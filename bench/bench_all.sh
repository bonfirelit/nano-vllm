#!/bin/bash
# Complete benchmark automation script
# Runs all performance tests and collects metrics

set -e

# Configuration
FP16_MODEL="${FP16_MODEL:-$HOME/huggingface/Qwen3-0.6B}"
AWQ_MODEL="${AWQ_MODEL:-$HOME/huggingface/Qwen3-0.6B-AWQ}"
RESULTS_DIR="./benchmark_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "======================================"
echo "Nano-vLLM Quantization Benchmark"
echo "======================================"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run a test
run_test() {
    local test_name="$1"
    local cmd="$2"

    echo "--------------------------------------"
    echo "Running: $test_name"
    echo "--------------------------------------"

    if eval "$cmd"; then
        echo "✓ $test_name completed"
    else
        echo "✗ $test_name failed"
    fi
    echo ""
}

# 1. Quick benchmark
run_test "Quick Benchmark" \
    "python quick_bench.py 2>&1 | tee '$RESULTS_DIR/quick_bench.log'"

# 2. Memory analysis
run_test "Memory Breakdown Analysis" \
    "python profile_quant.py --model '$FP16_MODEL' --mode analyze 2>&1 | tee '$RESULTS_DIR/memory_analysis.log'"

# 3. PyTorch Profiler (FP16)
if [ -d "$FP16_MODEL" ]; then
    run_test "PyTorch Profiler (FP16)" \
        "python profile_quant.py --model '$FP16_MODEL' --mode pytorch --output '$RESULTS_DIR/pytorch_fp16' 2>&1 | tee '$RESULTS_DIR/pytorch_fp16.log'"
fi

# 4. PyTorch Profiler (AWQ)
if [ -d "$AWQ_MODEL" ]; then
    run_test "PyTorch Profiler (AWQ)" \
        "python profile_quant.py --model '$AWQ_MODEL' --mode pytorch --output '$RESULTS_DIR/pytorch_awq' 2>&1 | tee '$RESULTS_DIR/pytorch_awq.log'"
fi

# 5. Generate summary report
cat > "$RESULTS_DIR/README.md" << EOF
# Benchmark Results

Generated: $(date)

## Test Configuration
- FP16 Model: $FP16_MODEL
- AWQ Model: $AWQ_MODEL
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

## Files
- \`quick_bench.log\` - Quick performance comparison
- \`memory_analysis.log\` - Memory breakdown analysis
- \`pytorch_fp16.log\` - PyTorch profiler output for FP16
- \`pytorch_awq.log\` - PyTorch profiler output for AWQ
- \`pytorch_fp16/\` - TensorBoard traces for FP16 (view with \`tensorboard --logdir=.\`)
- \`pytorch_awq/\` - TensorBoard traces for AWQ

## Viewing Results

### Quick Summary
\`\`\`bash
cat '$RESULTS_DIR/quick_bench.log'
\`\`\`

### TensorBoard
\`\`\`bash
tensorboard --logdir='$RESULTS_DIR'
\`\`\`

### Memory Analysis
\`\`\`bash
cat '$RESULTS_DIR/memory_analysis.log'
\`\`\`
EOF

echo "======================================"
echo "All benchmarks completed!"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "View summary with: cat $RESULTS_DIR/README.md"
echo "======================================"
