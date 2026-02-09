#!/usr/bin/env python3
"""
Benchmark script for FP16 models.

Measures:
1. Memory usage (peak GPU memory)
2. Throughput (tokens/second)
3. Latency
"""

import os
import time
import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams

# ============================================
# Configuration
# ============================================
MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

BATCH_SIZE = 32
MAX_INPUT_LEN = 256
MAX_OUTPUT_LEN = 256
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return allocated, reserved


def benchmark_fp16(model_path: str) -> dict:
    """
    Benchmark FP16 model.

    Returns dict with metrics:
    - peak_memory_mb: Peak GPU memory usage
    - load_time_s: Model loading time
    - throughput_tokens_s: Overall throughput
    """
    print(f"\n{'='*60}")
    print(f"Testing: FP16 Model")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    results = {
        'model_type': 'FP16',
        'path': model_path
    }

    # Clean up
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    llm = None
    try:
        # Measure load time and memory
        start_load = time.time()

        llm = LLM(
            model_path,
            enforce_eager=False,
            max_model_len=2048,
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
        # Cleanup
        if llm is not None:
            del llm

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return results


def print_summary(results: dict):
    """Print summary table."""
    print(f"\n{'='*60}")
    print(f"{'FP16 MODEL BENCHMARK RESULTS':^60}")
    print(f"{'='*60}")

    print(f"\nModel: {results['path']}")
    print(f"Load Time: {results['load_time_s']:.2f}s")
    print(f"Load Memory: {results['load_memory_mb']:.2f} MB")
    print(f"Throughput: {results['throughput_tokens_s']:.2f} tokens/s")
    print(f"Peak Memory: {results['peak_memory_mb']:.2f} MB")
    print(f"{'='*60}")


def main():
    """Run FP16 benchmark."""
    if not os.path.isdir(MODEL_PATH):
        print(f"Error: FP16 model not found at {MODEL_PATH}")
        return

    results = benchmark_fp16(MODEL_PATH)
    print_summary(results)


if __name__ == "__main__":
    main()
