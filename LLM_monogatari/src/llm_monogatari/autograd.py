"""一个标量自动求导引擎。

大框架的反向传播最终也要遵守同样的链式法则。这里每个 Value 只装一个
浮点数，因此很慢，却可以逐行追踪。
"""

from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    def __init__(
        self,
        data: float,
        children: Iterable["Value"] = (),
        operation: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._previous = tuple(children)
        self._operation = operation
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    @staticmethod
    def _as_value(other: float | "Value") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: float | "Value") -> "Value":
        other = self._as_value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def backward() -> None:
            self.grad += output.grad
            other.grad += output.grad

        output._backward = backward
        return output

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __mul__(self, other: float | "Value") -> "Value":
        other = self._as_value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def backward() -> None:
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = backward
        return output

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def __pow__(self, power: float) -> "Value":
        output = Value(self.data**power, (self,), f"**{power}")

        def backward() -> None:
            self.grad += power * (self.data ** (power - 1)) * output.grad

        output._backward = backward
        return output

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: float | "Value") -> "Value":
        return self + -self._as_value(other)

    def __rsub__(self, other: float | "Value") -> "Value":
        return self._as_value(other) - self

    def __truediv__(self, other: float | "Value") -> "Value":
        return self * (self._as_value(other) ** -1)

    def tanh(self) -> "Value":
        value = math.tanh(self.data)
        output = Value(value, (self,), "tanh")

        def backward() -> None:
            self.grad += (1.0 - value**2) * output.grad

        output._backward = backward
        return output

    def relu(self) -> "Value":
        output = Value(max(0.0, self.data), (self,), "relu")

        def backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * output.grad

        output._backward = backward
        return output

    def backward(self) -> None:
        ordered: list[Value] = []
        visited: set[int] = set()

        def visit(node: Value) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._previous:
                visit(parent)
            ordered.append(node)

        visit(self)
        self.grad = 1.0
        for node in reversed(ordered):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


class Neuron:
    """一个全连接神经元：tanh(w·x + b)。"""

    def __init__(self, weights: list[float], bias: float = 0.0) -> None:
        self.weights = [Value(weight, label=f"w{index}") for index, weight in enumerate(weights)]
        self.bias = Value(bias, label="b")

    def __call__(self, inputs: list[float]) -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError("输入维数必须等于权重数")
        activation = sum((weight * item for weight, item in zip(self.weights, inputs)), self.bias)
        return activation.tanh()

    def parameters(self) -> list[Value]:
        return [*self.weights, self.bias]
