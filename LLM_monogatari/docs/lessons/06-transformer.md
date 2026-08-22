# 第 6 课：把零件装成 Transformer

本项目最终使用 decoder-only Transformer。一次前向传播可画成：

```text
token ids
   ↓ 查表
token embedding + position information
   ↓
causal self-attention ── 残差 ── normalization
   ↓
feed-forward network ── 残差 ── normalization
   ↓ 重复多层
linear projection → vocabulary logits
```

不同模型会把归一化放在子层前或后，使用不同位置编码、激活函数、门控和注意力变体，但这条主线很稳定。

## 每个零件为什么存在

- embedding：把离散 token id 变成连续向量；
- 位置：self-attention 本身不自动知道先后顺序；
- attention：让每个位置按内容汇聚上下文；
- FFN：对每个位置的表示做非线性变换，通常先扩维再缩回；
- residual：让子层学习“增量”，也给梯度更直接的路径；
- normalization：控制各层数值尺度；
- output projection：把隐藏维度映回整个词表的 logits。

运行：

```bash
make lesson N=6
```

`src/llm_monogatari/tiny_transformer.py` 只做一次前向传播，所有矩阵是 Python 列表。它的随机权重会给出荒谬预测。结构规定了信息如何流动，训练才决定流动什么。

## 参数量从哪里来

以隐藏维度 d、词表 V 为例，embedding 约有 `V×d` 个参数；每层 Q/K/V/输出投影约 `4d²`；FFN 常占约 `8d²` 或更多。层数、宽度、词表与门控设计一起决定参数量。0.6B 意味着约六亿个可训练标量，不意味着六亿条可单独读取的知识。

4-bit 权重理论上每参数半字节，但运行内存不只存权重：还包括量化尺度、KV cache、临时激活、系统和 Python。训练还要保存梯度或适配器状态，因此不能用“参数量 × 位数”直接断言内存够不够。

## 从 Transformer 到“大”语言模型

2017 年原论文主要面向翻译。随后 decoder-only 的生成式预训练、encoder-only 的双向表示、以及 encoder-decoder 路线分别发展。规模扩大带来更广泛的少样本能力，但也把数据治理、计算资源、不可复现实验和评测污染推到中心。

“Transformer 理解语言吗”不是单凭一个结构图能回答的问题。先把可测问题说清楚更有用：它在未见过的名字/地点组合上能否遵守格式？能否保持指代？输出是否稳定？失败是 tokenizer、数据、容量还是采样造成？本仓库的评测就从这些小问题开始。

## 完成标准

不看上图重画一次；指出“模型结构、参数、训练数据、推理策略”各自改变什么；解释随机初始化的完整 Transformer 为什么仍不会写故事。

原始资料见 [参考资料](../reference/REFERENCES.md#注意力与-transformer)。
