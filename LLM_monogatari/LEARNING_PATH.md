# LLM_monogatari 学习路线

这是一张进度表，不是另一篇长讲义。一次只处理当前阶段；完成一个勾选一个。

## 当前建议：阶段 1

第一次学习请运行：

```bash
make start
```

然后阅读[第 0 课](docs/lessons/00-computer-and-python.md)。今天做到能修改一行文字并重新运行，就已经完成。

---

## 阶段 1：文字变成数字

目标：看懂模型真正接收的是 token id 和向量，不是屏幕上的文字。

| 顺序 | 阅读 | 动手 | 你应该看到 |
|---:|---|---|---|
| 0 | [电脑与 Python](docs/lessons/00-computer-and-python.md) | `make check` | Mac、Python、内存和磁盘检查 |
| 1 | [分词](docs/lessons/01-tokenization.md) | `make lesson N=1` | 日文字符编号与 BPE 合并顺序 |
| 2 | [向量与相似度](docs/lessons/02-vectors-and-similarity.md) | `make lesson N=2` | 词袋向量与两个余弦分数 |

- [ ] 我能说明 token 与 token id 的区别。
- [ ] 我能解释 BPE 是统计相邻频次，不是理解词义。
- [ ] 我能手算两个二维向量的点积。

完成信号：不用看讲义，也能说出“文字 → token → id → 向量”。

---

## 阶段 2：数字怎样学会预测

目标：把损失、梯度、概率、注意力和 Transformer 串成一条路径。

| 顺序 | 阅读 | 动手 | 你应该看到 |
|---:|---|---|---|
| 3 | [神经网络](docs/lessons/03-neural-networks.md) | `make lesson N=3` | loss 从约 4 下降，神经元学会 AND |
| 4 | [概率与生成](docs/lessons/04-probability-and-generation.md) | `make lesson N=4` | bigram 生成、交叉熵和困惑度 |
| 5 | [注意力](docs/lessons/05-attention.md) | `make lesson N=5` | 右上角为 0 的因果注意力矩阵 |
| 6 | [Transformer](docs/lessons/06-transformer.md) | `make lesson N=6` | embedding、hidden state、logits 的形状 |

- [ ] 我能画出“预测 → loss → 梯度 → 更新参数”的循环。
- [ ] 我知道 temperature 改变采样，不会补充知识。
- [ ] 我能解释 causal mask 为什么挡住未来 token。
- [ ] 我能画出 token → embedding → attention → FFN → logits。

完成信号：你能解释为什么结构完整但随机初始化的 Transformer 仍不会写故事。

---

## 阶段 3：在这台 Mac 上微调

目标：用自己能逐篇检查的数据训练 LoRA，而不是复制一个不透明的训练命令。

| 顺序 | 阅读 | 动手 | 你应该得到 |
|---:|---|---|---|
| 7 | [数据与微调](docs/lessons/07-data-and-finetuning.md) | 修改至少 5 篇故事；`make data` | 28 / 4 / 4 数据划分 |
| 8 | [本机训练](docs/lessons/08-local-training.md) | `make train-install`、`make train` | `adapters/LLM_monogatari-lora/` |
| 8b | 同一课的评测部分 | `make evaluate` | `runs/evaluation-*.md` 对照表 |

- [ ] 我逐篇读过准备训练的数据。
- [ ] 我能说明 adapter 为什么不是完整模型。
- [ ] 我保留了基座输出，没有只挑最好的一篇。
- [ ] 我记录了 validation loss 和最差案例。

完成信号：模型在未见过的人名/地点组合上，比基座更稳定地遵守目标格式。

---

## 阶段 4：网页与分享

目标：理解输入怎样从浏览器到本机模型，并能随时停止公网访问。

| 顺序 | 阅读 | 动手 | 你应该得到 |
|---:|---|---|---|
| 9 | [网页与分享](docs/lessons/09-web-and-sharing.md) | `make web` | `http://127.0.0.1:8000` |
| 9b | 同一课的隧道部分 | 另开终端 `make share` | 临时 `trycloudflare.com` URL |
| 10 | [历史与评述](docs/lessons/10-history-and-critique.md) | 写一页实验札记 | 对成果与局限的清楚判断 |

- [ ] 我知道 localhost、网页服务器和公网隧道的区别。
- [ ] 我能用 `Control + C` 停止网页和分享。
- [ ] 我没有公开真实姓名、住址、密钥或私人故事。
- [ ] 我能说出当前模型至少三个不能可靠完成的任务。

完成信号：朋友能用临时 URL 生成故事，而你知道请求在哪里计算、URL 何时失效。

---

## 暂停后不知道从哪里继续

只做三步：

1. 回到本页，找到第一个没有勾选的项目；
2. 打开该行的讲义；
3. 只运行该行的一条命令。

全部课程的单页索引在 [docs/lessons/README.md](docs/lessons/README.md)。概念忘了就查[小词典](docs/reference/GLOSSARY.md)，不用从头重学。
