import unittest

from llm_monogatari.tokenization import CharacterTokenizer, SimpleBPE, normalize


class CharacterTokenizerTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        tokenizer = CharacterTokenizer.train(["葵は鎌倉へ。"])
        text = "葵は鎌倉へ。"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_unknown_character(self) -> None:
        tokenizer = CharacterTokenizer.train(["葵"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("蓮")), "□")

    def test_nfkc_normalization(self) -> None:
        self.assertEqual(normalize("ＡＢＣ"), "ABC")


class BPETests(unittest.TestCase):
    def test_bpe_is_reversible(self) -> None:
        bpe = SimpleBPE.train(["物語を書く", "物語を読む"], num_merges=5)
        text = "物語を書く"
        self.assertEqual(bpe.decode(bpe.encode(text)), text)

    def test_frequent_pair_is_merged(self) -> None:
        bpe = SimpleBPE.train(["物語", "物語", "物語"], num_merges=1)
        self.assertEqual(bpe.merges[0], ("物", "語"))

    def test_empty_training_data_fails(self) -> None:
        with self.assertRaises(ValueError):
            SimpleBPE.train([])


if __name__ == "__main__":
    unittest.main()
