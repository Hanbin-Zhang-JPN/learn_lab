"""A GPT-like decoder Transformer with the attention math left visible."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 176
    n_embd: int = 96
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.10

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        causal = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape

        # One matrix multiplication makes Q, K and V, then we expose the split.
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_size).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.causal_mask[:time, :time], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = weights @ v

        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.proj(attended))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class StoryTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = token_ids.shape
        if time > self.config.block_size:
            raise ValueError(f"序列长度 {time} 超过 block_size={self.config.block_size}")
        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 0.85,
        top_k: int = 24,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            k = min(top_k, next_logits.size(-1))
            threshold = torch.topk(next_logits, k).values[:, [-1]]
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_id), dim=1)
            if bool((next_id == eos_id).all()):
                break
        return token_ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
