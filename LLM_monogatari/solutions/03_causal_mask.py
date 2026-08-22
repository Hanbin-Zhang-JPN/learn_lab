from llm_monogatari.attention import scaled_dot_product_attention

vectors = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
for causal in [True, False]:
    result = scaled_dot_product_attention(vectors, vectors, vectors, causal=causal)
    print("causal =", causal)
    for row in result.weights:
        print([round(value, 3) for value in row])
