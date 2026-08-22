# 小词典

| 词 | 本项目中的简短意思 |
|---|---|
| token | tokenizer 切出的处理单位，不一定是一个词 |
| vocabulary | token 与整数 id 的固定映射表 |
| embedding | token id 查到的可训练向量 |
| parameter | 训练会改变的数字，例如权重与偏置 |
| hyperparameter | 人设定的训练选择，例如学习率、rank、轮次 |
| logit | softmax 之前、尚未归一化的分数 |
| loss | 用一个数表示预测与目标的差异 |
| gradient | 参数微小增加时 loss 的局部变化率 |
| optimizer | 根据梯度更新参数的规则 |
| epoch | 训练数据大致完整看过一遍 |
| iteration / step | 一次或一次累积后的参数更新 |
| attention | 用 Q/K 匹配得到权重，再汇总 V |
| causal mask | 禁止当前位置读取未来 token 的遮罩 |
| context window | 单次可处理的最大 token 范围 |
| pretraining | 在大语料上学习一般语言分布 |
| fine-tuning | 在较小目标数据上继续调整行为 |
| LoRA | 用低秩增量矩阵适配部分权重 |
| quantization | 用更少比特近似保存或计算权重 |
| adapter | LoRA 训练得到的小参数文件，需配合基座 |
| inference | 参数固定后，用模型生成或预测 |
| overfitting | 训练题变好、未见题不再变好甚至变差 |
| perplexity | 平均交叉熵的指数；只宜在同设置下比较 |
