import unittest
from collections import Counter

from monogatari_llm.data import STYLES, normalize_style, records, story_prompt, training_text


class DataTests(unittest.TestCase):
    def test_records_are_reproducible(self) -> None:
        self.assertEqual(list(records(3, seed=9)), list(records(3, seed=9)))

    def test_conditions_appear_in_story(self) -> None:
        for record in records(20):
            self.assertIn(record["name"], record["story"])
            self.assertIn(record["place"], record["story"])
            self.assertIn(record["style"], STYLES)

    def test_prompt_starts_the_story(self) -> None:
        prompt = story_prompt("葵", "鎌倉", "文艺")
        self.assertEqual(prompt, "名前:葵\n場所:鎌倉\n作風:文芸\n物語:鎌倉で、葵は")

    def test_styles_are_balanced(self) -> None:
        counts = Counter(record["style"] for record in records(40))
        self.assertEqual(counts, Counter({style: 10 for style in STYLES}))

    def test_chinese_style_aliases(self) -> None:
        self.assertEqual(normalize_style("爱情"), "恋愛")
        self.assertEqual(normalize_style("文艺"), "文芸")
        self.assertEqual(normalize_style("幽默"), "ユーモア")
        self.assertEqual(normalize_style("恐怖"), "恐怖")

    def test_training_text_contains_style_condition(self) -> None:
        record = next(records(1))
        self.assertIn(f"作風:{record['style']}", training_text(record))


if __name__ == "__main__":
    unittest.main()
