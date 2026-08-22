"""第 3 课：反向传播让一个神经元学会 AND。"""

from llm_monogatari.autograd import Neuron, Value


training_data = [
    ([0.0, 0.0], -1.0),
    ([0.0, 1.0], -1.0),
    ([1.0, 0.0], -1.0),
    ([1.0, 1.0], 1.0),
]
neuron = Neuron([0.1, -0.2], bias=0.0)

for step in range(100):
    predictions = [neuron(inputs) for inputs, _ in training_data]
    loss = sum(
        ((prediction - target) ** 2 for prediction, (_, target) in zip(predictions, training_data)),
        Value(0.0),
    )

    for parameter in neuron.parameters():
        parameter.grad = 0.0
    loss.backward()
    for parameter in neuron.parameters():
        parameter.data -= 0.05 * parameter.grad

    if step in {0, 9, 49, 99}:
        print(f"step={step + 1:>3} loss={loss.data:.4f}")

print("\n训练后的预测：")
for inputs, target in training_data:
    print(f"{inputs} -> {neuron(inputs).data: .3f}（目标 {target:+.0f}）")
print("参数：", [round(value.data, 3) for value in neuron.parameters()])
