"""第 3 课：手写一个只有两个参数的神经元。"""


examples = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]
weight = -0.5
bias = 0.0
learning_rate = 0.05

print(f"开始：weight={weight:.3f}, bias={bias:.3f}")
for step in range(101):
    grad_weight = 0.0
    grad_bias = 0.0
    loss = 0.0
    for x, target in examples:
        prediction = weight * x + bias
        error = prediction - target
        loss += error * error
        grad_weight += 2 * error * x
        grad_bias += 2 * error

    count = len(examples)
    loss /= count
    weight -= learning_rate * grad_weight / count
    bias -= learning_rate * grad_bias / count
    if step % 20 == 0:
        print(f"step {step:3d} | loss={loss:.5f} | weight={weight:.3f} | bias={bias:.3f}")

print(f"\n学到的式子：y ≈ {weight:.2f}x + {bias:.2f}")
print("训练大模型仍是这个循环：预测 → 算误差 → 求梯度 → 微调参数。")

