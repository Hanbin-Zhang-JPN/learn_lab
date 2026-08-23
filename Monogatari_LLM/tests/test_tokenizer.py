import unittest

from monogatari_llm.tokenizer import CharTokenizer


class TokenizerTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        text = "鎌倉で、葵は歩いた。"
        tokenizer = CharTokenizer.build([text])
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(tokenizer.decode(ids), text)

    def test_unknown_character_uses_unk(self) -> None:
        tokenizer = CharTokenizer.build(["葵"])
        self.assertEqual(tokenizer.encode("凛"), [tokenizer.unk_id])


if __name__ == "__main__":
    unittest.main()

