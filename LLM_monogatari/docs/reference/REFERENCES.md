# 参考资料

优先列原始论文与项目官方文档。阅读顺序不按年份：先读课程，再带着具体问题查原文。论文里的数学不必一次看完，先读摘要、图和结论。

## 分词

- Philip Gage, 1994, *A New Algorithm for Data Compression*：BPE 压缩算法的早期公开描述。
- Mike Schuster & Kaisuke Nakajima, 2012, [Japanese and Korean Voice Search](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)：WordPiece 在日/韩语音搜索中的应用。
- Rico Sennrich et al., 2016, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：把 BPE 子词用于稀有词翻译。
- Taku Kudo & John Richardson, 2018, [SentencePiece](https://arxiv.org/abs/1808.06226)：从原始句子训练的语言无关子词系统。

## 向量与表示

- Thomas Landauer & Susan Dumais, 1997, [A Solution to Plato's Problem](https://doi.org/10.1037/0033-295X.104.2.211)：Latent Semantic Analysis 的系统论述。
- Yoshua Bengio et al., 2003, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)：联合学习分布式词表示与语言模型。
- Tomas Mikolov et al., 2013, [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)：word2vec 代表论文。
- Tolga Bolukbasi et al., 2016, [Man is to Computer Programmer as Woman is to Homemaker?](https://arxiv.org/abs/1607.06520)：词向量偏见研究，也可借此反思“去偏”定义。

## 神经网络与反向传播

- Warren McCulloch & Walter Pitts, 1943, [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://doi.org/10.1007/BF02478259)。
- Frank Rosenblatt, 1958, [The Perceptron](https://doi.org/10.1037/h0042519)。
- David Rumelhart, Geoffrey Hinton & Ronald Williams, 1986, [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0)。
- Sepp Hochreiter & Jürgen Schmidhuber, 1997, [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)。

## 概率语言模型

- Claude Shannon, 1948, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)：熵、信息与字符序列实验。
- Yoshua Bengio et al., 2003, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)。
- Ilya Sutskever et al., 2014, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)。

## 注意力与 Transformer

- Dzmitry Bahdanau, Kyunghyun Cho & Yoshua Bengio, 2014, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)。
- Ashish Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。
- Jacob Devlin et al., 2018, [BERT](https://arxiv.org/abs/1810.04805)。
- Tom Brown et al., 2020, [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)。

## 微调与量化

- Edward Hu et al., 2021, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)。
- Tim Dettmers et al., 2023, [QLoRA](https://arxiv.org/abs/2305.14314)。
- Apple ML Research, [MLX 源代码与说明](https://github.com/ml-explore/mlx)。
- Apple ML Research, [MLX-LM LoRA 官方说明](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)。命令会随版本变化，本项目固定 0.31.3。
- Qwen, [Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)；本项目运行的是 MLX 4-bit 转换版本。

## 运行与分享

- Apple ML Research, [MLX-LM 项目](https://github.com/ml-explore/mlx-lm)。
- Cloudflare, [Quick Tunnels 官方说明](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。官方明确把 Quick Tunnel 定位为测试/开发用途。

## 进一步的批判性阅读

- Emily Bender et al., 2021, [On the Dangers of Stochastic Parrots](https://doi.org/10.1145/3442188.3445922)：规模、数据文档与社会风险。
- Rishi Bommasani et al., 2021, [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)：基础模型的机会、风险与研究议程。
- Margaret Mitchell et al., 2019, [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)：为何模型应记录用途、评测与限制。
- Timnit Gebru et al., 2018, [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)：为什么数据也需要来源、动机与限制说明。
