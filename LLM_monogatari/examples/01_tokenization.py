"""第 1 课：同一段文字的字符分词与 BPE。"""

from llm_monogatari.tokenization import CharacterTokenizer, SimpleBPE


texts = [
    "葵は鎌倉を歩いた。",
    "葵は鎌倉で手紙を拾った。",
    "蓮は鎌倉を訪れた。",
]

characters = CharacterTokenizer.train(texts)
sentence = "葵は鎌倉を歩いた。"
ids = characters.encode(sentence)

print("原文：", sentence)
print("字符：", list(sentence))
print("编号：", ids)
print("还原：", characters.decode(ids))

bpe = SimpleBPE.train(texts, num_merges=8, min_frequency=2)
print("\nBPE 学到的合并顺序：")
for number, pair in enumerate(bpe.merges, start=1):
    print(f"{number:>2}. {pair[0]!r} + {pair[1]!r} -> {''.join(pair)!r}")
print("BPE 结果：", bpe.encode(sentence))
print("还原：", bpe.decode(bpe.encode(sentence)))
