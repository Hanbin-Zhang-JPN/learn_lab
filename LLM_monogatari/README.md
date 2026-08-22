# LLM_monogatari

从一段日文怎样变成 token 开始，最后在 MacBook Air M3 上训练一个日文短故事模型。

> 第一次打开时，不要依次点开所有文件。先在 VS Code 终端运行 `make start`，然后只看 [学习路线](LEARNING_PATH.md)。

## 现在一目了然

| 部分 | 当前状态 | 最终看得见的结果 |
|---|---|---|
| 原理课程 | 已准备好 | 6 个可运行实验：分词、向量、神经网络、生成、注意力、Transformer |
| 故事数据 | 已准备好 | 36 篇可逐篇阅读的日文故事卡，自动划分为 28 / 4 / 4 |
| 模型训练 | 等你学到第 8 课 | `adapters/LLM_monogatari-lora/` 微调参数 |
| 分享网页 | 演示模式已验证 | 输入日文人名和地名，得到日文故事；可生成临时 URL |

“已准备好”表示代码、文档和测试已经完成，不表示模型已经替你训练。模型下载与微调特意留到学完原理后再做。

## 第一次只做这一件事

```bash
cd /Users/hanbin_zhang/github_codex/LLM_monogatari
make start
```

你会先看到环境准备信息，随后看到一句日文如何被字符分词、编号、还原，以及 BPE 如何合并常见片段。这是整个项目的第一个可见结果。

如果还不知道怎样在 VS Code 打开终端，先看[零基础操作说明](docs/guide/START_HERE.md)。

接着打开：

1. [学习路线：我现在在哪一步](LEARNING_PATH.md)
2. [课程目录：每课解决什么问题](docs/lessons/README.md)
3. [第 0 课：电脑与 Python](docs/lessons/00-computer-and-python.md)

其余文件暂时不用看。

## 四个阶段

| 阶段 | 学什么 | 运行什么 | 完成后应能看到/解释 |
|---|---|---|---|
| 1. 文字变成数字 | 第 0–2 课 | `make lesson N=1`、`N=2` | token id、BPE 合并、余弦相似度 |
| 2. 数字学会预测 | 第 3–6 课 | `make lesson N=3` 到 `N=6` | loss 下降、bigram 生成、因果注意力矩阵、logits |
| 3. 本机微调 | 第 7–8 课 | `make data`、`make train`、`make evaluate` | LoRA adapter 与基座/微调对照表 |
| 4. 网页分享 | 第 9–10 课 | `make web`，另开终端 `make share` | 本地网页和临时公网 URL |

详细的勾选项、完成标准和暂停后如何继续，都在 [LEARNING_PATH.md](LEARNING_PATH.md)。

## 想先看看最终网页

演示模式不下载模型，使用明确标注的固定故事模板：

```bash
make demo
```

浏览器访问 `http://127.0.0.1:8000`，输入例如：

```text
人物名：葵
場所：鎌倉
```

按 `Control + C` 停止。演示结果只证明网页流程正常，不代表微调效果。

## 学到第 8 课以后

```bash
make train-install   # 首次联网安装 MLX-LM
make data            # 检查并划分故事卡
make train           # 首次联网下载约 351 MB 的模型，然后训练
make evaluate        # 保存基座与微调结果对照
make web             # 使用 adapter 启动网页
```

分享时保持 `make web` 运行，另开终端执行 `make share`。Mac 必须保持联网和唤醒；临时 URL 不适合作为长期公开服务。

## 当你需要找文件时

| 目录 | 什么时候才需要打开 |
|---|---|
| `docs/lessons/` | 当前正在学习某一课 |
| `examples/` | 想修改课堂小实验 |
| `exercises/` | 学完一课后做短练习 |
| `src/llm_monogatari/` | 讲义明确要求阅读某段实现时 |
| `data/source/` | 第 7 课开始审校自己的故事数据时 |
| `config/` | 第 8 课理解训练参数时 |
| `web/` | 第 9 课学习网页时 |
| `docs/reference/` | 查词、查论文、模型限制或透明度时 |
| `tests/`、`scripts/` | 排错或想理解工程流程时 |

## 为什么仍然强调“少黑盒”

分词、向量、softmax、反向传播、注意力和教学 Transformer 都用 Python 标准库手写。最终训练才使用 Apple Silicon 上的开源 MLX/MLX-LM。预训练权重仍然无法逐参数解释，临时公网隧道也是外部服务；这些边界没有被隐藏，详见[透明度账本](docs/reference/TRANSPARENCY.md)。

模型与数据的适用范围见[模型说明卡](docs/reference/MODEL_CARD.md)和[数据说明卡](docs/reference/DATA_CARD.md)。代码采用 MIT License，外部模型与工具保留各自许可。
