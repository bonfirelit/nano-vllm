import os
import argparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="NanoVLLM 对话交互脚本")
    # 模型路径
    parser.add_argument("--model_path", type=str, default="~/huggingface/Qwen3-0.6B/", help="模型本地路径")
    # TP 并行大小
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel 数量")
    # 是否强制使用 Eager 模式
    parser.add_argument("--enforce_eager", action="store_true", help="是否强制使用 Eager 模式")
    # 采样温度
    parser.add_argument("--temperature", type=float, default=0.6, help="生成温度 (0-1)")
    # 最大生成长度
    parser.add_argument("--max_tokens", type=int, default=256, help="最大生成长度")
    
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = os.path.expanduser(args.model_path)

    # 1. 初始化模型与分词器
    print(f"正在加载模型: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp_size)

    # 2. 配置生成参数
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # 3. 进入对话循环
    history = []
    print("\n--- 进入对话模式 (输入 'exit' 或 'quit' 退出) ---")
    
    while True:
        user_input = input("\n用户: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("对话结束。")
            break
        
        if not user_input:
            continue

        # 将当前用户输入加入历史
        history.append({"role": "user", "content": user_input})

        # 使用 chat_template 格式化输入
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 4. 执行推理 (注意：generate 接收列表，返回列表)
        outputs = llm.generate([prompt], sampling_params)
        
        # 获取回复文本
        response_text = outputs[0]['text']
        
        print(f"AI: {response_text}")

        # 将模型回复加入历史，实现多轮对话
        history.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()