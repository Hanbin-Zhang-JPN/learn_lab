# 课程目录

每一课只回答一个核心问题，并配一个可见结果。按编号学习，不需要预先打开源码目录。

| 课 | 核心问题 | 实验命令 | 看到什么就算运行成功 |
|---:|---|---|---|
| [0](00-computer-and-python.md) | 文件、程序、Python 环境是什么？ | `make check` | 系统检查逐项显示“通过/注意” |
| [1](01-tokenization.md) | 日文怎样变成 token id？ | `make lesson N=1` | 字符编号、还原文字、BPE 合并 |
| [2](02-vectors-and-similarity.md) | 文字怎样进入向量空间？ | `make lesson N=2` | 0.6 与 0.0 两个相似度示例 |
| [3](03-neural-networks.md) | 参数怎样从错误中改变？ | `make lesson N=3` | loss 下降、AND 预测接近目标 |
| [4](04-probability-and-generation.md) | 下一个 token 怎样被预测和抽样？ | `make lesson N=4` | bigram 文本、交叉熵、困惑度 |
| [5](05-attention.md) | 一个位置怎样读取前文？ | `make lesson N=5` | causal attention 矩阵 |
| [6](06-transformer.md) | 前面的零件怎样组成 Transformer？ | `make lesson N=6` | 四组矩阵形状与随机预测 |
| [7](07-data-and-finetuning.md) | LoRA 改了什么，数据为何更重要？ | `make data` | 28 / 4 / 4 数据划分 |
| [8](08-local-training.md) | 怎样在 M3 16GB 上安全训练？ | `make train` | adapter checkpoint 和 loss 日志 |
| [9](09-web-and-sharing.md) | 本地模型怎样变成可访问的页面？ | `make web` | 浏览器中的日文故事页面 |
| [10](10-history-and-critique.md) | 这些技术怎样发展，又该怎样评价？ | 写实验札记 | 能说明证据、失败和黑盒边界 |

学习时固定使用这个循环：

```text
先预测会发生什么 → 运行实验 → 改一个地方 → 再运行 → 写一句结论
```

完成一课后回到[学习路线](../../LEARNING_PATH.md)勾选，不要连续阅读很多课却不运行实验。
