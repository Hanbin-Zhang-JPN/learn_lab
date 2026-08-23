# Monogatari_LLM

这是一个三小时左右的练习：在 MacBook Air M3 上看懂一个很小的日文语言模型，并从随机权重开始训练它。

目标很具体——输入一个日文人名、地点和故事作风（恋爱 / 文艺 / 幽默 / 恐怖），写出一段短故事。文字怎样变成 token，token 怎样变成向量，注意力怎样工作，训练循环怎样修改参数，都能在仓库里找到。训练只用 NumPy 和 PyTorch，不调用模型 API，也不下载别人的权重。

模型约有 40 万个参数。它的文笔和常识都有限，但大小刚好适合第一次把整个过程走通。

## 你最后会得到什么

- 一条约 3 小时、没有岔路的学习路线；
- 一个自己写的数据集生成器，不需要外部语料或 API；
- 一个从零训练的字符级 Transformer；
- 一个输入日文姓名、地名和作风后生成短故事的网页；
- 一个可发到微信的 URL。临时分享时你的 Mac 需要保持开机；发布后则不需要。

## 第一次打开，只做这四步

1. 在 VS Code 里打开整个 `Monogatari_LLM` 文件夹。
2. 打开菜单 **Terminal → New Terminal**。
3. 复制下面两行，每行按一次回车：

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

4. 安装结束后，继续输入：

```bash
source .venv/bin/activate
python lessons/00_check.py
```

如果最后看到 `准备完成，可以从第 1 课开始。`，就打开 [docs/00_从这里开始.md](docs/00_从这里开始.md)。

## 三小时路线

按编号前进，不需要挑选章节。

| 时间 | 阅读 | 运行 | 带走一个概念 |
|---:|---|---|---|
| 00:00–00:15 | `00_从这里开始` | `00_check.py` | 训练到底在做什么 |
| 00:15–00:40 | `01_分词` | `01_tokenizer.py` | token 与词表 |
| 00:40–01:05 | `02_向量与相似度` | `02_vectors.py` | 向量、点积、余弦相似度 |
| 01:05–01:30 | `03_神经网络` | `03_neuron.py` | 参数、损失、梯度下降 |
| 01:30–01:35 | 休息 | — | 离开屏幕五分钟 |
| 01:35–02:10 | `04_注意力` | `04_attention.py` | Q、K、V 与因果遮罩 |
| 02:10–02:40 | `05_Transformer` | `05_transformer.py` | 多层组件怎样连起来 |
| 02:40–03:00 | `06_训练与分享` | `06_data.py` | 数据、训练、推理的闭环 |

每个脚本都能单独运行，会在终端里一步步打印。看不懂一行时，先改一个数字再运行；这里不要求你先学完整套 Python。

## 三小时之后：训练自己的模型

仍在已经激活 `.venv` 的终端中运行：

```bash
python -m monogatari_llm.train
```

默认设置是为 16GB Apple Silicon 准备的：自动使用 MPS，最多运行 85 分钟，并保存验证损失最好的版本。通常不必跑满；合成数据的规律比较简单。按 `Control + C` 可以提前停止，已经保存的最佳版本不会丢失。

固定时长无法诚实保证固定文笔。如果 85 分钟结束时验证 loss 仍在稳定下降，可以从零重新做一轮较长实验：

```bash
python -m monogatari_llm.train --steps 4500 --time-limit-minutes 120
```

不要一开始就用长版本；先完成默认训练并读几条结果。

训练结束后试一段故事：

```bash
python -m monogatari_llm.generate --name 葵 --place 鎌倉 --style 文芸
```

结果不够好时，先读 [docs/07_效果不理想时.md](docs/07_效果不理想时.md)，不要盲目把模型放大。

## 打开网页

把刚训练的权重转换为浏览器可读的普通二进制文件，然后启动网页：

```bash
python -m monogatari_llm.export_web
npm install
npm run dev
```

浏览器访问 `http://localhost:3000`。推理发生在浏览器里，姓名、地名与作风不会发送给模型 API。

## 分享给微信好友

最少步骤的临时分享方式：

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:3000
```

终端会显示一个 `https://...trycloudflare.com` 地址，把它复制到微信即可。关闭终端或让 Mac 休眠后，地址就会失效。Cloudflare Tunnel 是此流程中唯一不可避免的外部中继；模型、权重和生成计算仍由你控制。

若要长期有效的地址，把 `public/model.bin` 和 `public/model-config.json` 一起发布。这个项目已经包含可部署的网站骨架；在 Codex 中说“发布 Monogatari_LLM 网站”即可更新固定链接。

## 仓库地图

```text
Monogatari_LLM/
├── docs/                 依次阅读的课程
├── lessons/              每课一个可运行实验
├── src/monogatari_llm/   数据、分词、模型、训练、生成、导出
├── tests/                小而快的自检
├── app/                  分享网页
├── public/               浏览器读取的模型文件（训练后生成）
├── artifacts/            本机训练检查点（不会提交到 Git）
└── data/                 本机生成的训练数据（不会提交到 Git）
```

## 设计取舍

- **字符级分词**：日文不天然用空格分词。字符级方法不是效率最高的，但最容易看清输入与编号的对应关系。
- **合成训练数据**：没有版权下载和清洗黑盒，所有句子部件都能在 `data.py` 里找到；同时也意味着模型的想象力受这些部件限制。
- **手写注意力**：没有使用 `nn.MultiheadAttention`；Q/K/V、缩放、遮罩和 softmax 都直接写在 `model.py`。
- **PyTorch 只负责张量与自动求导**：如果连矩阵乘法和求导都重写，三小时会变成三周。
- **浏览器纯 TypeScript 推理**：不依赖远程推理服务。代码比使用现成推理库长一些，但每一步可检查。

## 安全与隐私

只加载自己训练得到的 `artifacts/monogatari.pt`。PyTorch 检查点不是应该从陌生人处随便下载并打开的文件。这个小模型没有内容审核，不应拿真实隐私信息、仇恨或伤害性内容做公开演示。

## 参考入口

- [PyTorch：macOS 本地安装](https://docs.pytorch.org/get-started/locally/)
- [PyTorch：MPS 后端](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Mikolov 等：Word2Vec（2013）](https://arxiv.org/abs/1301.3781)
- [Vaswani 等：Attention Is All You Need（2017）](https://arxiv.org/abs/1706.03762)
