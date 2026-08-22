# 第 0 课：先认识工作台

这一课不讲模型。目标是让你知道命令在哪里运行、代码从哪里开始，以及出错时怎样保住现场。

## 文件、程序与进程

文件是磁盘上的内容，例如 `examples/01_tokenization.py`。Python 是读取这种文件并执行指令的程序。执行中的程序叫进程。按 `Control + C` 通常是在告诉当前进程“请停止”，不是删除文件。

终端中的命令由“程序名 + 参数”组成：

```bash
PYTHONPATH=src .venv/bin/python examples/01_tokenization.py
```

- `PYTHONPATH=src`：告诉 Python 也去 `src` 找我们写的模块；
- `.venv/bin/python`：使用本项目的 Python；
- `examples/01_tokenization.py`：要执行的文件。

`make lesson N=1` 只是把这条较长的命令包装成易记的短命令。打开 `Makefile` 可以看见它，没有隐藏操作。

## 先会读五种 Python 形状

```python
name = "葵"                         # 变量：给一个值起名字
tokens = ["葵", "は", "歩く"]      # 列表：有顺序的一组值
card = {"name": "葵", "place": "鎌倉"}  # 字典：键和值

def greet(person):                  # 函数：可重复使用的一段步骤
    return "こんにちは、" + person

for token in tokens:                # 循环：对每个元素做一次
    print(token)
```

现在无需背诵。遇到不认识的语法，先问三件事：输入是什么、输出是什么、哪一行改变了数据。

## 安全边界

本课程不需要 `sudo`。模型下载、Python 包和训练结果都应留在用户目录或仓库目录。运行网上命令前先检查它是否会删除文件、上传文件或写系统目录。

Git 的作用是记录改动，不等于云备份。`git status` 只读取状态；`git diff` 显示变化；`git add` 与 `git commit` 才记录一个版本；`git push` 才会把内容送到远端。

## 本课实验

```bash
make check
make test
```

然后打开 `examples/01_tokenization.py`，把第一句中的 `鎌倉` 改成 `京都`，运行 `make lesson N=1`。如果出现未知字符方框，这正好说明“词表之外”是什么。

## 完成标准

你能说明 `.venv` 为什么不该提交到 Git；能用 `Control + C` 停止程序；能分清“改了文件”和“运行了文件”。
