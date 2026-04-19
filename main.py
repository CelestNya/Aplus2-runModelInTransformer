import signal
import warnings
import torch
import threading
from transformers import TextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer

# 屏蔽无关紧要的 warnings
warnings.filterwarnings("ignore", message="triton not found")
warnings.filterwarnings("ignore", message="torch_dtype")
warnings.filterwarnings("ignore", message="Setting `pad_token_id`")
warnings.filterwarnings("ignore", message="The fast path is not available")
warnings.filterwarnings("ignore", message="_check_is_size")
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

# TODO: 替换为你的模型路径
MODEL_PATH = "D:/Resources/Models/Qwen3.5-0.8B"

# 量化选项: "none", "4bit", "8bit"
QUANTIZATION = "none"

# 上下文压缩设置
MAX_CONTEXT_MESSAGES = 16      # 超过此轮数触发压缩
MIN_KEEP_MESSAGES = 4          # 压缩时保留最近 N 条消息不Summarize
COMPRESSION_SUMMARY_PREFIX = "[旧对话摘要]"  # 摘要的前缀标记

# 全局停止标志
_stop_event = threading.Event()


def _handle_sigint(signum, frame):
    """Ctrl+C 优雅处理：打印提示并设置停止标志"""
    del signum, frame
    print("\n  [Interrupt] 正在停止，请等待当前回复生成完毕...")
    _stop_event.set()


signal.signal(signal.SIGINT, _handle_sigint)


def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    quantization_config = None
    if QUANTIZATION == "4bit":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        print("  -> 4bit quantization enabled")
    elif QUANTIZATION == "8bit":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("  -> 8bit quantization enabled")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def _summarize_messages(model, tokenizer, messages, max_summary_tokens=128):
    """将旧对话压缩为摘要"""
    to_summarize = messages[:-MIN_KEEP_MESSAGES] if len(messages) > MIN_KEEP_MESSAGES else []
    if not to_summarize:
        return None

    conv_text = ""
    for msg in to_summarize:
        role = msg["role"].upper()
        conv_text += f"{role}: {msg['content']}\n"

    summary_prompt = (
        f"请将以下对话精炼为简短摘要，保留关键信息、用户需求和已完成结论：\n"
        f"{conv_text}\n简要摘要："
    )

    inputs = tokenizer(summary_prompt, return_tensors="pt", truncation=True).to(model.device)
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=max_summary_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    summary = tokenizer.decode(summary_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return summary


def _compress_context(model, tokenizer, messages):
    """压缩上下文：旧对话Summarize为一条摘要消息"""
    summary = _summarize_messages(model, tokenizer, messages)
    if summary is None:
        return messages

    system_msgs = [m for m in messages if m["role"] == "system"]
    recent_msgs = messages[-MIN_KEEP_MESSAGES:]

    compressed = system_msgs[:]
    compressed.append({"role": "system", "content": f"{COMPRESSION_SUMMARY_PREFIX} {summary}"})
    compressed.extend(recent_msgs)

    print(f"\n  [Context compressed: {len(messages)} -> {len(compressed)} messages]")
    return compressed


def main():
    model, tokenizer = load_model()
    print("Model loaded! (Ctrl+C 优雅退出)\n")

    messages = []

    while not _stop_event.is_set():
        try:
            user_input = input("You: ")
        except (EOFError, OSError):
            break
        if user_input.lower() in ("quit", "exit"):
            break
        if _stop_event.is_set():
            break

        messages.append({"role": "user", "content": user_input})

        if len(messages) > MAX_CONTEXT_MESSAGES:
            messages = _compress_context(model, tokenizer, messages)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        gen_thread = threading.Thread(
            target=model.generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.5,
                "pad_token_id": tokenizer.pad_token_id,
            },
        )
        gen_thread.start()

        print("Bot: ", end="", flush=True)
        response = ""
        try:
            for text_chunk in streamer:
                if _stop_event.is_set():
                    break
                print(text_chunk, end="", flush=True)
                response += text_chunk
        except GeneratorExit:
            pass
        finally:
            gen_thread.join()

        print()
        if _stop_event.is_set():
            break

        messages.append({"role": "assistant", "content": response})

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
