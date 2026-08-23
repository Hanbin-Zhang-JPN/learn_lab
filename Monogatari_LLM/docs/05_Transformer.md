# 05｜Transformer：把零件装成语言模型（30 分钟）

现在把前四课连起来：

```text
token id
  ↓  token embedding + position embedding
向量序列
  ↓  LayerNorm → 多头因果注意力 → 残差相加
  ↓  LayerNorm → 前馈网络 → 残差相加
  ↓  重复 3 层
  ↓  线性投影为整个词表的 logits
下一个字符的概率
```

## 每块的工作

- **位置向量**：注意力本身不知道先后，位置 embedding 告诉它第几个字符。
- **多头**：把 96 维分成 4 组，每组可以形成不同的关联方式；这不保证每头都有能用自然语言命名的职责。
- **残差**：把新结果加回旧结果，为信息和梯度保留直路。
- **LayerNorm**：稳定每个位置的数值尺度。
- **前馈网络**：注意力负责位置之间交换，前馈层负责每个位置内部变换。
- **logits**：未归一化分数；softmax 后才是下一字符概率。

本项目是 **decoder-only** Transformer：只做“看左边、猜右边”。2018 年 GPT 展示了先做通用语言模型预训练、再适配任务的路线；之后规模迅速增长。但规模定律并没有取消数据质量、评测和部署成本。

历史入口：[Improving Language Understanding by Generative Pre-Training（2018）](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)。

## 运行

```bash
python lessons/05_transformer.py
```

随机模型也能输出 logits，但猜测无意义。这是重要区别：**架构规定它怎样计算，训练数据决定参数学到什么。**

打开 `model.py`，按顺序只看这五个类：`CausalSelfAttention`、`FeedForward`、`TransformerBlock`、`StoryTransformer.forward`、`generate`。

完成标志：你能沿上面的箭头，说出输入最终怎样变成下一 token。然后进入 `06_训练与分享.md`。

