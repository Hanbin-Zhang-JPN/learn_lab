"""第 2 课：词袋向量与余弦相似度。"""

from llm_monogatari.vectors import bag_of_words, cosine_similarity, make_vocabulary


documents = {
    "海边": list("海風船青い"),
    "港口": list("港風船白い"),
    "森林": list("森木鳥緑の"),
}
vocabulary = make_vocabulary(list(documents.values()))
vectors = {name: bag_of_words(tokens, vocabulary) for name, tokens in documents.items()}

print("词表：", vocabulary)
for name, vector in vectors.items():
    print(f"{name}：{vector}")

print("\n相似度（1 最接近，0 表示正交）：")
print("海边 vs 港口：", round(cosine_similarity(vectors["海边"], vectors["港口"]), 3))
print("海边 vs 森林：", round(cosine_similarity(vectors["海边"], vectors["森林"]), 3))
print("\n提醒：这个数字只反映我们选择的表示法，不是句子的客观意义。")
