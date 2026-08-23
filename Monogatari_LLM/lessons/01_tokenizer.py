"""第 1 课：亲眼看见文字变成编号。"""

from monogatari_llm.tokenizer import CharTokenizer


texts = ["葵は鎌倉へ行った。", "凛は京都へ行った。"]
tokenizer = CharTokenizer.build(texts)
sentence = "葵は京都へ行った。"
ids = tokenizer.encode(sentence, add_bos=True, add_eos=True)

print("训练文字：", texts)
print("词表大小：", tokenizer.vocab_size)
print("字符 → 编号：")
for char, token_id in zip(sentence, tokenizer.encode(sentence)):
    print(f"  {char!r} → {token_id}")
print("加上开头、结尾标记：", ids)
print("再解码：", tokenizer.decode(ids))

unknown = "葵は東京へ行った。"
print("\n词表没见过“東”“京”以外的新字符时，会出现 <unk> 编号：")
print(unknown, "→", tokenizer.encode(unknown))
print("请试试：把上面的句子换成自己的姓名。")

