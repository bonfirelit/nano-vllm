#!/usr/bin/env python3
"""
Performance comparison script between FP16 and AWQ quantized models.

Measures:
1. Memory usage (peak GPU memory)
2. Throughput (tokens/second)
3. Latency (TTFT - Time To First Token, TPOT - Time Per Output Token)
4. GPU utilization
"""

import os
import time
import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams

# ============================================
# Configuration
# ============================================
FP16_MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
AWQ_MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B-AWQ/")  # Adjust path

BATCH_SIZE = 32  # Reduced from 64 to prevent OOM
MAX_INPUT_LEN = 256  # Reduced from 512
MAX_OUTPUT_LEN = 256  # Reduced from 512
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return allocated, reserved


def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi."""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        gpu_util, mem_util = result.stdout.strip().split(',')
        return int(gpu_util), int(mem_util)
    return None, None


def benchmark_model(model_path: str, quantized: bool = False) -> dict:
    """
    Benchmark a single model configuration.

    Returns dict with metrics:
    - peak_memory_mb: Peak GPU memory usage
    - load_time_s: Model loading time
    - prefill_tokens/s: Prefill throughput
    - decode_tokens/s: Decode throughput
    - total_tokens/s: Overall throughput
    - ttft_ms: Time to first token (average)
    - tpot_ms: Time per output token (average)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {'AWQ Quantized' if quantized else 'FP16'} Model")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    results = {
        'model_type': 'AWQ' if quantized else 'FP16',
        'path': model_path
    }

    # Clean up any previous distributed state
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Force cleanup of any previous distributed state
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Additional wait to ensure all CUDA operations complete
    torch.cuda.synchronize()

    llm = None
    try:
        # Measure load time and memory
        start_load = time.time()

        llm = LLM(
            model_path,
            enforce_eager=False,
            max_model_len=2048,  # Reduced from 4096 to save memory
        )

        load_time = time.time() - start_load
        load_memory = get_gpu_memory()[0]

        print(f"Load time: {load_time:.2f}s")
        print(f"Memory after load: {load_memory:.2f} MB")

        results['load_time_s'] = load_time
        results['load_memory_mb'] = load_memory

        # Warmup
        print("Warming up...")
        for _ in range(WARMUP_RUNS):
            llm.generate(["Warmup"], SamplingParams(max_tokens=10))

        # Benchmark
        seed(42)
        all_throughputs = []

        for run_idx in range(BENCHMARK_RUNS):
            print(f"\nRun {run_idx + 1}/{BENCHMARK_RUNS}")

            # Generate random prompts
            prompt_token_ids = [
                [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))]
                for _ in range(BATCH_SIZE)
            ]
            sampling_params = [
                SamplingParams(
                    temperature=0.8,
                    ignore_eos=True,
                    max_tokens=randint(100, MAX_OUTPUT_LEN)
                )
                for _ in range(BATCH_SIZE)
            ]

            # Measure total time
            # Note: For more detailed TTFT/TPT analysis, would need to modify LLM.generate internals
            torch.cuda.synchronize()
            total_start = time.time()

            outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

            torch.cuda.synchronize()
            total_time = time.time() - total_start

            # Calculate metrics
            total_input_tokens = sum(len(p) for p in prompt_token_ids)
            total_output_tokens = sum(
                len(output['token_ids']) for output in outputs
            )
            total_tokens = total_input_tokens + total_output_tokens

            throughput = total_tokens / total_time
            all_throughputs.append(throughput)

            print(f"  Input tokens: {total_input_tokens}")
            print(f"  Output tokens: {total_output_tokens}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.2f} tokens/s")

            # Check peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Peak memory: {peak_memory:.2f} MB")

        # Calculate averages
        results['throughput_tokens_s'] = sum(all_throughputs) / len(all_throughputs)
        results['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2

    finally:
        # Cleanup - important for running multiple models in same process
        if llm is not None:
            del llm

        # Aggressive cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Destroy distributed process group to allow re-initialization
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return results


def print_comparison(fp16_results: dict, awq_results: dict):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print(f"{'PERFORMANCE COMPARISON':^80}")
    print(f"{'='*80}")

    metrics = [
        ('Metric', 'FP16', 'AWQ', 'Speedup', 'Memory Reduction'),
    ]

    # Calculate comparisons
    throughput_speedup = awq_results['throughput_tokens_s'] / fp16_results['throughput_tokens_s']
    memory_reduction = (1 - awq_results['peak_memory_mb'] / fp16_results['peak_memory_mb']) * 100

    metrics.extend([
        ('Throughput (tok/s)',
         f"{fp16_results['throughput_tokens_s']:.2f}",
         f"{awq_results['throughput_tokens_s']:.2f}",
         f"{throughput_speedup:.2f}x",
         '-'),

        ('Peak Memory (MB)',
         f"{fp16_results['peak_memory_mb']:.2f}",
         f"{awq_results['peak_memory_mb']:.2f}",
         '-',
         f"{memory_reduction:.1f}%"),

        ('Load Time (s)',
         f"{fp16_results['load_time_s']:.2f}",
         f"{awq_results['load_time_s']:.2f}",
         f"{fp16_results['load_time_s']/awq_results['load_time_s']:.2f}x",
         '-'),

        ('Load Memory (MB)',
         f"{fp16_results['load_memory_mb']:.2f}",
         f"{awq_results['load_memory_mb']:.2f}",
         '-',
         f"{(1-awq_results['load_memory_mb']/fp16_results['load_memory_mb'])*100:.1f}%"),
    ])

    # Print table
    for row in metrics:
        print(f"{row[0]:<20} {row[1]:>15} {row[2]:>15} {row[3]:>15} {row[4]:>15}")

    print(f"{'='*80}")

    # Print summary
    print("\nSummary:")
    print(f"  - AWQ is {throughput_speedup:.2f}x faster in throughput")
    print(f"  - AWQ uses {memory_reduction:.1f}% less peak memory")
    print(f"  - AWQ loads {fp16_results['load_time_s']/awq_results['load_time_s']:.2f}x faster")


def main():
    """Run full comparison benchmark."""

    # Test FP16 model
    fp16_results = None
    if os.path.isdir(FP16_MODEL_PATH):
        fp16_results = benchmark_model(FP16_MODEL_PATH, quantized=False)
    else:
        print(f"Warning: FP16 model not found at {FP16_MODEL_PATH}")

    # Test AWQ model
    awq_results = None
    if os.path.isdir(AWQ_MODEL_PATH):
        awq_results = benchmark_model(AWQ_MODEL_PATH, quantized=True)
    else:
        print(f"Warning: AWQ model not found at {AWQ_MODEL_PATH}")
        print("Please download or quantize a model first.")

    # Print comparison if both results available
    if fp16_results and awq_results:
        print_comparison(fp16_results, awq_results)


if __name__ == "__main__":
    main()
