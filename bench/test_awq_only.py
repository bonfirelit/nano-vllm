#!/usr/bin/env python3
"""
Test only AWQ model to avoid OOM from running both models.
Use this if bench_quant.py runs out of memory.
"""

import os
import gc
import time
import torch
from nanovllm import LLM, SamplingParams


def test_awq_only(model_path: str):
    """Test AWQ model alone."""
    print(f"\n{'='*60}")
    print(f"Testing AWQ Model Only")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    # Clear everything before starting
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print(f"\nInitial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Load model
    load_start = time.time()
    llm = LLM(
        model_path,
        enforce_eager=False,
        max_model_len=2048,  # Reduce if still OOM
        gpu_memory_utilization=0.85,  # Leave some headroom
    )
    load_time = time.time() - load_start

    load_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"\nLoad time: {load_time:.2f}s")
    print(f"Memory after load: {load_mem:.2f} MB")

    # Warmup
    print("\nWarming up...")
    llm.generate(["Warmup"], SamplingParams(max_tokens=10))

    # Test with progressively larger batches
    configs = [
        {"batch": 4, "tokens": 64, "name": "Small"},
        {"batch": 8, "tokens": 128, "name": "Medium"},
        {"batch": 16, "tokens": 128, "name": "Large"},
    ]

    results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Test: {config['name']} (batch={config['batch']}, max_tokens={config['tokens']})")
        print(f"{'='*60}")

        prompts = ["Tell me a short story about AI."] * config['batch']
        sampling_params = SamplingParams(max_tokens=config['tokens'], temperature=0.8)

        torch.cuda.synchronize()
        start = time.time()

        try:
            outputs = llm.generate(prompts, sampling_params)

            torch.cuda.synchronize()
            elapsed = time.time() - start

            total_tokens = sum(len(out['token_ids']) for out in outputs)
            throughput = total_tokens / elapsed
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2

            print(f"✓ Success!")
            print(f"  Generated: {total_tokens} tokens")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {throughput:.2f} tokens/s")
            print(f"  Peak memory: {peak_mem:.2f} MB")

            results.append({
                **config,
                'throughput': throughput,
                'peak_memory': peak_mem,
                'success': True
            })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ OOM! Batch size {config['batch']} too large")
                results.append({**config, 'success': False, 'error': 'OOM'})
                # Clear and try next config
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise e

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Load time: {load_time:.2f}s")
    print(f"Load memory: {load_mem:.2f} MB")

    for r in results:
        if r['success']:
            print(f"\n{r['name']}: {r['throughput']:.2f} tok/s, {r['peak_memory']:.2f} MB")
        else:
            print(f"\n{r['name']}: OOM")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Set your AWQ model path
    awq_path = os.path.expanduser("~/huggingface/Qwen3-4B-AWQ/")

    # Or use environment variable
    # awq_path = os.environ.get("AWQ_MODEL", awq_path)

    test_awq_only(awq_path)
