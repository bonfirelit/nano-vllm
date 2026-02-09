#!/usr/bin/env python3
"""
Quick benchmark script for comparing FP16 vs AWQ models.
Simple version that focuses on key metrics.
"""

import os
import time
import gc
import torch
from nanovllm import LLM, SamplingParams


def quick_benchmark(model_path: str, name: str):
    """Run a quick benchmark on a model."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")

    # Clean up any previous distributed state
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    llm = None
    try:
        # Load model and measure time/memory
        load_start = time.time()
        llm = LLM(model_path, enforce_eager=False, max_model_len=2048)
        load_time = time.time() - load_start

        load_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"Load time: {load_time:.2f}s, Memory: {load_mem:.2f} MB")

        # Warmup
        llm.generate(["Warmup"], SamplingParams(max_tokens=10))

        # Benchmark - reduce batch size if needed
        prompts = ["Tell me a short story about AI."] * 8  # Reduced from 16
        sampling_params = SamplingParams(max_tokens=64, temperature=0.8)  # Reduced from 128

        torch.cuda.synchronize()
        start = time.time()

        outputs = llm.generate(prompts, sampling_params)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate metrics
        total_tokens = sum(len(out['token_ids']) for out in outputs)
        throughput = total_tokens / elapsed
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"Generated {total_tokens} tokens in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.2f} tokens/s")
        print(f"Peak memory: {peak_mem:.2f} MB")

    finally:
        # Cleanup
        if llm is not None:
            del llm

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Destroy distributed process group to allow re-initialization
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return {
        'name': name,
        'load_time': load_time,
        'load_memory': load_mem,
        'throughput': throughput,
        'peak_memory': peak_mem,
    }


def main():
    # Update these paths
    fp16_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    awq_path = os.path.expanduser("~/huggingface/Qwen3-0.6B-AWQ/")

    results = []

    # Test FP16
    if os.path.isdir(fp16_path):
        results.append(quick_benchmark(fp16_path, "FP16"))
    else:
        print(f"FP16 model not found at {fp16_path}")

    # Test AWQ
    if os.path.isdir(awq_path):
        results.append(quick_benchmark(awq_path, "AWQ"))
    else:
        print(f"AWQ model not found at {awq_path}")
        print("\nTip: To quantize a model, use:")
        print("  pip install autoawq")
        print("  python -m awq.entry --model_path <fp16_path> --w_bit 4 --q_group_size 128")

    # Print comparison
    if len(results) == 2:
        fp16, awq = results
        print(f"\n{'='*50}")
        print("COMPARISON")
        print(f"{'='*50}")
        print(f"Throughput:  {fp16['throughput']:.1f} -> {awq['throughput']:.1f} tok/s ({awq['throughput']/fp16['throughput']:.2f}x)")
        print(f"Peak Memory: {fp16['peak_memory']:.0f} -> {awq['peak_memory']:.0f} MB ({(1-awq['peak_memory']/fp16['peak_memory'])*100:.1f}% reduction)")
        print(f"Load Time:   {fp16['load_time']:.2f} -> {awq['load_time']:.2f}s ({fp16['load_time']/awq['load_time']:.2f}x)")


if __name__ == "__main__":
    main()
