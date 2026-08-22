"""第 5 课：观察因果注意力矩阵。"""

from llm_monogatari.attention import scaled_dot_product_attention


tokens = ["葵", "は", "本", "を", "読む"]
queries = [
    [1.0, 0.0],
    [0.8, 0.2],
    [0.1, 0.9],
    [0.2, 0.8],
    [0.7, 0.7],
]
keys = queries
values = [
    [1.0, 0.0, 0.0],  # 人物信息
    [0.0, 0.0, 0.2],  # 助词信息
    [0.0, 1.0, 0.0],  # 物体信息
    [0.0, 0.0, 0.2],
    [0.0, 0.0, 1.0],  # 动作信息
]

result = scaled_dot_product_attention(queries, keys, values, causal=True)
print("每行是当前 token 对各位置分配的注意力：")
print(" " * 8 + " ".join(f"{token:>5}" for token in tokens))
for token, row in zip(tokens, result.weights):
    print(f"{token:>5} : " + " ".join(f"{weight:5.2f}" for weight in row))

print("\n上三角为 0：生成当前位置时不能偷看未来 token。")
print("‘読む’汇总后的 value：", [round(value, 3) for value in result.outputs[-1]])
