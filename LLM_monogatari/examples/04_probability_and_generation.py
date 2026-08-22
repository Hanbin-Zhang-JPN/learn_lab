"""第 4 课：bigram 只看一个字符，也可以继续生成。"""

from llm_monogatari.bigram import BigramLanguageModel
from llm_monogatari.probability import perplexity, softmax


corpus = [
    "葵は海を見た。",
    "葵は月を見た。",
    "蓮は海へ行った。",
    "凛は月を見上げた。",
]
model = BigramLanguageModel(smoothing=0.1)
model.train(corpus)

print("logits [2, 1, 0] 经 softmax：", [round(x, 3) for x in softmax([2, 1, 0])])
print("“葵”之后的候选：")
for token, probability in sorted(model.distribution("葵"), key=lambda item: -item[1])[:5]:
    print(f"  {token!r}: {probability:.3f}")

print("\n不同随机种子生成的文字（它们不一定像日文）：")
for seed in range(3):
    print(seed, model.generate(seed=seed, max_characters=28, temperature=0.8))

loss = model.negative_log_likelihood("葵は海を見た。")
print(f"\n平均交叉熵：{loss:.3f}，困惑度：{perplexity(loss):.2f}")
