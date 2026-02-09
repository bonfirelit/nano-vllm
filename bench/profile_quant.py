#!/usr/bin/env python3
"""
Detailed profiling script using PyTorch profiler and Nsight tools.

Usage:
1. Basic PyTorch profiling:
   python profile_quant.py --mode pytorch

2. Nsight Systems (timeline analysis):
   nsys profile -o profile_output python profile_quant.py --mode nsys

3. Nsight Compute (kernel analysis):
   ncu --set full -o profile_output python profile_quant.py --mode ncu
"""

import argparse
import os
import time
import torch
from nanovllm import LLM, SamplingParams


def profile_with_pytorch(llm, output_dir: str = "./profile_results"):
    """Profile using PyTorch profiler."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a simple prompt for profiling
    prompts = ["Explain quantum computing in detail."] * 8
    sampling_params = SamplingParams(max_tokens=256, temperature=0.8)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir)
    ) as prof:
        llm.generate(prompts, sampling_params)

    # Print summary
    print("\n" + "="*60)
    print("PyTorch Profiler Summary")
    print("="*60)

    print("\nTop 10 CUDA operations by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\nTop 10 operations by memory usage:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    print(f"\nTrace saved to {output_dir}")
    print("View with: tensorboard --logdir={output_dir}")


def profile_with_nsys(llm, num_steps: int = 10):
    """
    Run workload for Nsight Systems profiling.

    The actual profiling is done by wrapping this script with nsys:
    nsys profile -o output.qdrep python profile_quant.py --mode nsys
    """
    prompts = ["Write a short story about AI."] * 16
    sampling_params = SamplingParams(max_tokens=128, temperature=0.8)

    # Warmup
    print("Warming up...")
    llm.generate(["Warmup"], SamplingParams(max_tokens=10))

    # Synchronize and start measured region
    torch.cuda.synchronize()

    for i in range(num_steps):
        print(f"Step {i+1}/{num_steps}")
        llm.generate(prompts, sampling_params, use_tqdm=False)

    torch.cuda.synchronize()


def profile_with_ncu(llm):
    """
    Run workload for Nsight Compute profiling.

    The actual profiling is done by wrapping this script with ncu:
    ncu --set full -o output python profile_quant.py --mode ncu
    """
    # Use a smaller workload for kernel-level analysis
    prompts = ["Explain machine learning."] * 4
    sampling_params = SamplingParams(max_tokens=64, temperature=0.8)

    # Warmup
    llm.generate(["Warmup"], SamplingParams(max_tokens=10))

    # Run profiled region
    torch.cuda.synchronize()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()

    # Print basic stats
    total_output = sum(len(out['token_ids']) for out in outputs)
    print(f"\nGenerated {total_output} tokens across {len(prompts)} sequences")


def analyze_memory_breakdown(model_path: str):
    """Analyze memory usage breakdown by component."""
    from nanovllm.config import Config

    print("\n" + "="*60)
    print("Memory Breakdown Analysis")
    print("="*60)

    config = Config(model=model_path)

    # Get model config
    hf_config = config.hf_config
    num_layers = hf_config.num_hidden_layers
    hidden_size = hf_config.hidden_size
    num_attention_heads = hf_config.num_attention_heads
    intermediate_size = getattr(hf_config, 'intermediate_size', hidden_size * 4)
    vocab_size = hf_config.vocab_size

    print(f"\nModel Architecture:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Vocab size: {vocab_size}")

    # Estimate memory for different components (FP16)
    params_per_layer = (
        # Attention
        4 * hidden_size * hidden_size +  # qkv_proj, o_proj
        # MLP
        2 * hidden_size * intermediate_size +  # gate_up_proj
        1 * intermediate_size * hidden_size +  # down_proj
        # Layer norms (FP32)
        6 * hidden_size
    )

    total_params = (
        params_per_layer * num_layers +
        hidden_size * vocab_size  # embedding
    )

    fp16_memory = total_params * 2 / (1024**2)  # 2 bytes per param

    # Estimate AWQ memory (4-bit)
    awq_memory = total_params * 0.5 / (1024**2)  # 0.5 bytes per param

    # Add scales and zeros (for AWQ)
    group_size = 128
    num_groups = total_params // group_size
    scales_zeros_memory = num_groups * 2 * 2 / (1024**2)  # fp16 scales + zeros

    awq_total_memory = awq_memory + scales_zeros_memory

    # KV cache estimate (per token)
    kv_cache_per_token = (
        2 * num_layers *  # 2 for K and V
        2 * hidden_size / num_attention_heads  # 2 for kv_head_dim
    )
    kv_cache_per_token_mb = kv_cache_per_token * 2 / (1024**2)

    print(f"\nEstimated Memory Breakdown:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  FP16 Weights: {fp16_memory:.2f} MB")
    print(f"  AWQ Weights: {awq_memory:.2f} MB")
    print(f"  AWQ Scales+Zeros: {scales_zeros_memory:.2f} MB")
    print(f"  AWQ Total: {awq_total_memory:.2f} MB")
    print(f"  Compression ratio: {fp16_memory/awq_total_memory:.2f}x")

    print(f"\nKV Cache (per token): {kv_cache_per_token_mb:.3f} MB")
    print(f"KV Cache (1024 tokens): {kv_cache_per_token_mb*1024:.2f} MB")
    print(f"KV Cache (max_len={config.max_model_len}): {kv_cache_per_token_mb*config.max_model_len:.2f} MB (per batch of 1)")

    # Activation memory (estimate)
    activation_memory = (
        BATCH_SIZE := 32
    )
    seq_len = 512
    activation_mb = (
        activation_memory * seq_len * hidden_size * 4 / (1024**2)
    )
    print(f"\nActivation Memory (batch={activation_memory}, seq={seq_len}): {activation_mb:.2f} MB")


def compute_intensity_analysis():
    """
    Analyze arithmetic intensity of different operations.

    Arithmetic Intensity = FLOPs / Bytes
    Higher is better for GPU utilization.
    """
    print("\n" + "="*60)
    print("Arithmetic Intensity Analysis")
    print("="*60)

    # For a typical transformer layer:
    # - Linear layer: 2 * M * N FLOPs, reads M*N weights, reads N inputs, writes M outputs
    # - For M=N=hidden_size=4096, batch=32, seq=512

    batch = 32
    seq = 512
    hidden = 4096

    # MatMul: QK^T (attention)
    qk_flops = 2 * batch * num_heads := 32 * seq * seq * hidden // num_heads := 32
    qk_bytes = (
        3 * batch * seq * hidden * 2 +  # Q, K (read)
        batch * num_heads * seq * seq * 2  # Output (write)
    )
    qk_intensity = qk_flops / qk_bytes

    # MatMul: Softmax * V
    av_flops = 2 * batch * num_heads * seq * seq * (hidden // num_heads)
    av_bytes = (
        batch * num_heads * seq * seq * 2 +  # Attention weights (read)
        batch * seq * hidden * 2 +  # V (read)
        batch * seq * hidden * 2  # Output (write)
    )
    av_intensity = av_flops / av_bytes

    # Linear layer
    linear_flops = 2 * batch * seq * hidden * hidden
    linear_bytes = (
        batch * seq * hidden * 2 +  # Input (read)
        hidden * hidden * 2 +  # Weights (read)
        batch * seq * hidden * 2  # Output (write)
    )
    linear_intensity = linear_flops / linear_bytes

    print(f"\nOperation              | FLOPs        | Bytes        | Intensity (FLOP/Byte)")
    print(f"-----------------------|--------------|--------------|----------------------")
    print(f"QK^T (Attention)       | {qk_flops/1e9:.2f} GFLOPs | {qk_bytes/1e9:.2f} GB     | {qk_intensity:.2f}")
    print(f"Attn * V               | {av_flops/1e9:.2f} GFLOPs | {av_bytes/1e9:.2f} GB     | {av_intensity:.2f}")
    print(f"Linear Layer           | {linear_flops/1e9:.2f} GFLOPs | {linear_bytes/1e9:.2f} GB     | {linear_intensity:.2f}")

    # AWQ effect on intensity
    awq_linear_bytes = (
        batch * seq * hidden * 2 +  # Input (read)
        hidden * hidden * 0.5 +  # 4-bit Weights (read)
        (hidden // 128) * hidden * 2 * 2 +  # Scales + Zeros (read)
        batch * seq * hidden * 2  # Output (write)
    )
    awq_linear_intensity = linear_flops / awq_linear_bytes

    print(f"\nWith AWQ Quantization:")
    print(f"Linear Layer (AWQ)      | {linear_flops/1e9:.2f} GFLOPs | {awq_linear_bytes/1e9:.2f} GB     | {awq_linear_intensity:.2f}")
    print(f"\nIntensity improvement: {awq_linear_intensity/linear_intensity:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Profile nano-vLLM models")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
                        help="Path to model")
    parser.add_argument("--mode", type=str, choices=["pytorch", "nsys", "ncu", "analyze"],
                        default="pytorch", help="Profiling mode")
    parser.add_argument("--output", type=str, default="./profile_results",
                        help="Output directory for traces")

    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_memory_breakdown(args.model)
        compute_intensity_analysis()
        return

    # Load model
    print(f"Loading model from {args.model}...")
    llm = LLM(args.model, enforce_eager=True, max_model_len=2048)

    if args.mode == "pytorch":
        profile_with_pytorch(llm, args.output)
    elif args.mode == "nsys":
        profile_with_nsys(llm)
    elif args.mode == "ncu":
        profile_with_ncu(llm)


if __name__ == "__main__":
    main()
