PYTHON := .venv/bin/python
ENV := PYTHONPATH=src

.PHONY: help start setup check test data lesson demo web train-install train evaluate share clean-generated

help:
	@echo "make start          第一次学习：准备环境并运行分词实验"
	@echo "make setup          创建本项目的 Python 环境"
	@echo "make check          检查 Mac、Python、磁盘和可选工具"
	@echo "make lesson N=1     运行第 N 个原理实验（1 到 6）"
	@echo "make test           运行不需要模型的全部测试"
	@echo "make data           从人工故事卡生成训练/验证/测试集"
	@echo "make demo           启动不加载模型的网页演示"
	@echo "make train-install  安装 MLX-LM 训练依赖（首次需联网）"
	@echo "make train          开始 LoRA 微调（首次下载约 351 MB 模型）"
	@echo "make web            用微调结果启动本地网页"
	@echo "make share          把已启动的网页临时分享为 URL"

start: setup
	@echo ""
	@echo "========== 第一个可见结果：文字 → token =========="
	$(ENV) $(PYTHON) examples/01_tokenization.py
	@echo ""
	@echo "下一步只需打开：LEARNING_PATH.md"

setup:
	./scripts/setup.sh

check:
	$(PYTHON) scripts/check_system.py

test:
	$(ENV) $(PYTHON) -m unittest discover -s tests -v

data:
	$(ENV) $(PYTHON) scripts/prepare_data.py

lesson:
	$(ENV) $(PYTHON) examples/0$(N)_*.py

demo:
	$(ENV) $(PYTHON) -m llm_monogatari.web --demo

train-install:
	$(PYTHON) -m pip install -r requirements-train.txt

train: data
	./scripts/train.sh

evaluate:
	$(ENV) $(PYTHON) scripts/evaluate_stories.py

web:
	$(ENV) $(PYTHON) -m llm_monogatari.web

share:
	./scripts/share.sh

clean-generated:
	find data/processed -name '*.jsonl' -delete
