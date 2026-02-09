#!/usr/bin/env python3
"""
Run FP16 and AWQ benchmarks sequentially and compare results.
Avoids OOM by running each benchmark in a separate process.
"""

import os
import subprocess
import sys
import json
import tempfile


def run_benchmark(script_path: str) -> dict:
    """Run a benchmark script and capture its output."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"Error: {script_path} failed with exit code {result.returncode}")
        return None

    # Parse output to extract metrics
    return parse_output(result.stdout, script_path)


def parse_output(output: str, script_path: str) -> dict:
    """Parse benchmark output to extract key metrics."""
    metrics = {'script': os.path.basename(script_path)}

    # Common patterns to search for
    patterns = {
        'load_time_s': r'Load Time:\s*([\d.]+)s',
        'load_memory_mb': r'Load Memory:\s*([\d.]+)\s*MB',
        'throughput_tokens_s': r'Throughput:\s*([\d.]+)\s*tokens/s',
        'peak_memory_mb': r'Peak Memory:\s*([\d.]+)\s*MB',
    }

    import re
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def print_comparison(fp16_metrics: dict, awq_metrics: dict):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print(f"{'FP16 vs AWQ COMPARISON':^80}")
    print(f"{'='*80}")

    if not fp16_metrics or not awq_metrics:
        print("Error: Missing metrics for comparison")
        return

    print(f"\n{'Metric':<20} {'FP16':>15} {'AWQ':>15} {'Ratio':>15}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")

    # Load Time
    if 'load_time_s' in fp16_metrics and 'load_time_s' in awq_metrics:
        fp16_val = fp16_metrics['load_time_s']
        awq_val = awq_metrics['load_time_s']
        ratio = fp16_val / awq_val if awq_val > 0 else 0
        print(f"{'Load Time (s)':<20} {fp16_val:>15.2f} {awq_val:>15.2f} {ratio:>15.2f}x")

    # Load Memory
    if 'load_memory_mb' in fp16_metrics and 'load_memory_mb' in awq_metrics:
        fp16_val = fp16_metrics['load_memory_mb']
        awq_val = awq_metrics['load_memory_mb']
        reduction = (1 - awq_val / fp16_val) * 100 if fp16_val > 0 else 0
        print(f"{'Load Memory (MB)':<20} {fp16_val:>15.2f} {awq_val:>15.2f} {-reduction:>14.1f}%")

    # Throughput
    if 'throughput_tokens_s' in fp16_metrics and 'throughput_tokens_s' in awq_metrics:
        fp16_val = fp16_metrics['throughput_tokens_s']
        awq_val = awq_metrics['throughput_tokens_s']
        ratio = awq_val / fp16_val if fp16_val > 0 else 0
        print(f"{'Throughput (tok/s)':<20} {fp16_val:>15.2f} {awq_val:>15.2f} {ratio:>15.2f}x")

    # Peak Memory
    if 'peak_memory_mb' in fp16_metrics and 'peak_memory_mb' in awq_metrics:
        fp16_val = fp16_metrics['peak_memory_mb']
        awq_val = awq_metrics['peak_memory_mb']
        reduction = (1 - awq_val / fp16_val) * 100 if fp16_val > 0 else 0
        print(f"{'Peak Memory (MB)':<20} {fp16_val:>15.2f} {awq_val:>15.2f} {-reduction:>14.1f}%")

    print(f"{'='*80}")

    # Summary
    print("\nSummary:")
    if 'throughput_tokens_s' in fp16_metrics and 'throughput_tokens_s' in awq_metrics:
        speedup = awq_metrics['throughput_tokens_s'] / fp16_metrics['throughput_tokens_s']
        print(f"  - AWQ is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} in throughput")

    if 'peak_memory_mb' in fp16_metrics and 'peak_memory_mb' in awq_metrics:
        reduction = (1 - awq_metrics['peak_memory_mb'] / fp16_metrics['peak_memory_mb']) * 100
        print(f"  - AWQ uses {reduction:.1f}% {'less' if reduction > 0 else 'more'} peak memory")


def main():
    """Run both benchmarks and compare results."""
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    fp16_script = os.path.join(bench_dir, "bench_fp16.py")
    awq_script = os.path.join(bench_dir, "bench_awq.py")

    # Check if scripts exist
    if not os.path.exists(fp16_script):
        print(f"Error: {fp16_script} not found")
        return 1

    if not os.path.exists(awq_script):
        print(f"Error: {awq_script} not found")
        return 1

    # Run benchmarks sequentially
    fp16_metrics = run_benchmark(fp16_script)
    awq_metrics = run_benchmark(awq_script)

    # Print comparison
    if fp16_metrics and awq_metrics:
        print_comparison(fp16_metrics, awq_metrics)

        # Save results to JSON
        results = {
            'fp16': fp16_metrics,
            'awq': awq_metrics,
        }
        output_file = os.path.join(bench_dir, "benchmark_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
