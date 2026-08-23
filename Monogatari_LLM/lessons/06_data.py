"""第 6 课：查看模型真正会读到的训练样本。"""

from random import Random

from monogatari_llm.data import records, training_text


example = next(records(count=1))
print("JSON 记录：")
print(example)
print("\n送进模型的连续文本：")
print(training_text(example))

print("\n相同姓名、地点和作风也能组合出另一条故事：")
from monogatari_llm.data import make_story
print(make_story(Random(7), example["name"], example["place"], example["style"]))
print("\n下一步运行：python -m monogatari_llm.train")
