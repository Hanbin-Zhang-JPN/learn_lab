"""练习：比较两种分词的 token 数。把 TODO 换成你的代码。"""

from llm_monogatari.tokenization import SimpleBPE

texts = ["物語を書く。", "短い物語を読む。", "物語の種を探す。"]
bpe = SimpleBPE.train(texts, num_merges=6)
sentence = "短い物語を書く。"

character_count = 0  # TODO: 用 len(...) 数字符
bpe_count = 0  # TODO: 先 bpe.encode(sentence)，再数长度

print("字符 token 数：", character_count)
print("BPE token 数：", bpe_count)
