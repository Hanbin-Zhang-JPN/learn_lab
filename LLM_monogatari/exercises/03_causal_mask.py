"""练习：把 causal 改成 False，圈出矩阵中发生变化的位置。"""

from llm_monogatari.attention import scaled_dot_product_attention

vectors = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
result = scaled_dot_product_attention(vectors, vectors, vectors, causal=True)

for row in result.weights:
    print([round(value, 3) for value in row])
