import unittest

from llm_monogatari.autograd import Neuron, Value


class AutogradTests(unittest.TestCase):
    def test_chain_rule(self) -> None:
        x = Value(3.0)
        y = (x * x + 2 * x + 1).tanh()
        y.backward()
        expected = (1 - y.data**2) * 8.0
        self.assertAlmostEqual(x.grad, expected)

    def test_shared_node_accumulates_gradient(self) -> None:
        x = Value(2.0)
        y = x * x + x
        y.backward()
        self.assertAlmostEqual(x.grad, 5.0)

    def test_neuron_dimension_check(self) -> None:
        neuron = Neuron([1.0, 2.0])
        with self.assertRaises(ValueError):
            neuron([1.0])


if __name__ == "__main__":
    unittest.main()
