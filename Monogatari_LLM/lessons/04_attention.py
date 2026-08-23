"""第 4 课：用小列表完成一次单头注意力。"""

from math import exp, sqrt


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(values: list[float]) -> list[float]:
    largest = max(values)
    exps = [exp(value - largest) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


# 把三个字符想成三张卡片。Q 问“我在找什么”，K 写“我是什么”，V 是内容。
labels = ["雨", "猫", "駅"]
query = [0.8, 0.2]
keys = [[0.7, 0.3], [0.2, 0.9], [0.6, 0.1]]
values = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]

scores = [dot(query, key) / sqrt(len(query)) for key in keys]
weights = softmax(scores)
answer = [sum(weight * value[i] for weight, value in zip(weights, values)) for i in range(2)]

print("注意力分数：")
for label, score, weight in zip(labels, scores, weights):
    print(f"  {label}: score={score:.3f}, softmax 后权重={weight:.3f}")
print("\n按权重混合 V，得到：", [round(value, 3) for value in answer])
print("\n语言模型还会遮住右侧未来字符；否则训练时就等于偷看答案。")

