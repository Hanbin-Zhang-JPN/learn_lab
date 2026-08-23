"""第 2 课：不用任何库，算一次余弦相似度。"""

from math import sqrt


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def length(vector: list[float]) -> float:
    return sqrt(dot(vector, vector))


def cosine(a: list[float], b: list[float]) -> float:
    return dot(a, b) / (length(a) * length(b))


# 三个方向暂时人为命名为：[自然，城市，安静]
vectors = {
    "森": [0.9, 0.1, 0.8],
    "海": [0.8, 0.1, 0.6],
    "駅": [0.1, 0.9, 0.2],
}

print("每个词现在是一组三维坐标：")
for word, vector in vectors.items():
    print(word, vector)

print("\n余弦相似度越接近 1，方向越接近：")
print("森 ↔ 海:", round(cosine(vectors["森"], vectors["海"]), 3))
print("森 ↔ 駅:", round(cosine(vectors["森"], vectors["駅"]), 3))
print("\n真正训练时，这些坐标不是人填写的，而是梯度下降学出来的。")

