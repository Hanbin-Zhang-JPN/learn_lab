"""第 5 课：让一个随机初始化的 Transformer 走一次前向传播。"""

import torch

from monogatari_llm.model import ModelConfig, StoryTransformer
from monogatari_llm.tokenizer import CharTokenizer


text = "名前:葵\n場所:鎌倉\n作風:文芸\n物語:鎌倉で、葵は"
tokenizer = CharTokenizer.build([text])
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=64,
    n_embd=16,
    n_head=4,
    n_layer=1,
    dropout=0.0,
)
model = StoryTransformer(config)
token_ids = torch.tensor([tokenizer.encode(text, add_bos=True)])
logits, _ = model(token_ids)

print("输入形状 [批次, 字符数]：", tuple(token_ids.shape))
print("输出形状 [批次, 字符数, 词表]：", tuple(logits.shape))
print("参数数量：", model.parameter_count())
next_id = int(logits[0, -1].argmax())
print("随机模型猜的下一个字符：", repr(tokenizer.decode([next_id])))
print("\n现在的猜测没有意义；训练就是反复纠正这些 logits。")
