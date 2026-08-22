"""第 6 课：一次完整但未训练的 Transformer 前向传播。"""

from llm_monogatari.tiny_transformer import TinyDecoder
from llm_monogatari.tokenization import CharacterTokenizer


text = "葵は鎌倉へ行く。"
tokenizer = CharacterTokenizer.train([text])
ids = tokenizer.encode(text)
decoder = TinyDecoder(vocabulary_size=len(tokenizer.id_to_token), dimension=6, seed=7)
trace = decoder.forward(ids)

print("文字：", text)
print("token ids：", ids)
for name, matrix in trace.items():
    print(f"{name:>18}: {len(matrix)} × {len(matrix[0])}")

last_logits = trace["logits"][-1]
best_id = max(range(len(last_logits)), key=last_logits.__getitem__)
print("\n随机参数给出的下一个 token：", tokenizer.decode([best_id]))
print("这不是合理预测：结构搭好了，但权重还没从数据中学习。")
