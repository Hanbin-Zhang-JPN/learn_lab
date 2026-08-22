"""练习：预测哪两句更相似，再用词袋余弦检查。"""

from llm_monogatari.vectors import bag_of_words, cosine_similarity, make_vocabulary

sentences = [list("猫が窓で眠る"), list("猫が椅子で眠る"), list("船が港を出る")]
vocabulary = make_vocabulary(sentences)
vectors = [bag_of_words(sentence, vocabulary) for sentence in sentences]

# TODO: 分别计算 (0, 1)、(0, 2)、(1, 2) 的相似度。
print("先在纸上写下预测，再补完这里。")
