from llm_monogatari.tokenization import SimpleBPE

texts = ["物語を書く。", "短い物語を読む。", "物語の種を探す。"]
bpe = SimpleBPE.train(texts, num_merges=6)
sentence = "短い物語を書く。"

character_count = len(sentence)
bpe_count = len(bpe.encode(sentence))

print("字符 token 数：", character_count)
print("BPE token 数：", bpe_count)
