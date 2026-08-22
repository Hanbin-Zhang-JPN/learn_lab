# 贡献与个人修改

这是学习仓库，最有价值的贡献是让一个概念更容易被验证，而不是加入更多框架。

修改前运行：

```bash
make test
git status
```

修改后再运行 `make test`。新增原理函数应同时增加一个小例子和至少一个测试；新增故事卡必须通过 `make data`，并在数据卡记录来源与局限。

提交信息可写成朴素动作，例如：

```text
补充 BPE 合并顺序说明
修正 causal mask 的示例
增加 12 篇人工审校故事
```

不要提交 `.venv`、模型权重、adapter、运行日志中的私人输入或任何密钥。若公开到 GitHub，先逐项检查 `git status` 与 `git diff --staged`。
