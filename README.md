# Aplus2-runModelInTransformer

学习 transformers 库过程中写的小工具，用来在本地加载模型并对话。

## 主要内容

- 模型加载：学习 `AutoModelForCausalLM` 和 `AutoTokenizer` 的基本用法
- 流式输出：用 `TextIteratorStreamer` 实现打字效果
- 量化入门：尝试 bitsandbytes 的 4bit / 8bit 量化
- 上下文管理：对话轮次过多时，用 Summarize 的方式压缩历史

## 模型

默认使用 Qwen3.5-0.8B，需自行下载并修改 `MODEL_PATH`。

## 依赖

```
torch
transformers
accelerate
bitsandbytes
```

## 运行

```bash
uv sync
uv run python main.py
```

## 说明

这个项目主要是我自己用来熟悉 transformers 推理流程的，代码结构比较简单，没有做太多错误处理。

