# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight implementation of vLLM built from scratch in ~1,200 lines of Python. It provides fast offline inference comparable to vLLM with a clean, readable architecture. The project supports Qwen2/Qwen3 models and their MoE variants.

## Development Commands

### Installation
```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

### Model Download
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### Running Examples
```bash
# Basic usage example
python example.py

# Chat interface
python chat.py

# Performance benchmark
python bench.py
```

### Testing
No formal test suite exists. Validation is done through `example.py` and `bench.py`.

## Architecture

### Core Components

**Entry Point**: `nanovllm/llm.py` - The `LLM` class is a thin wrapper around `LLMEngine`.

**Inference Engine** (`nanovllm/engine/`):
- `llm_engine.py`: Main orchestration with multiprocessing for tensor parallelism
- `scheduler.py`: Batches and schedules requests for prefill/decode phases
- `model_runner.py`: Executes model across tensor-parallel processes using shared memory
- `block_manager.py`: Manages KV cache blocks
- `sequence.py`: Represents ongoing generation requests

**Models** (`nanovllm/models/`):
- `qwen2.py`, `qwen3.py`: Base model implementations
- `qwen2_moe.py`, `qwen3_moe.py`: Mixture-of-Experts variants
- Models are selected via `ModelRunner.model_dict` based on `hf_config.model_type`

**Layers** (`nanovllm/layers/`):
- Custom implementations with tensor parallelism support
- `attention.py`, `linear.py`: Core neural network components
- `fuse_moe/`: Fused MoE operations with Triton kernels
- `sampler.py`: Token sampling logic

### Execution Flow

1. **Initialization** (`LLMEngine.__init__`):
   - Spawns N-1 worker processes for tensor parallelism (rank 1 to N-1)
   - Main process holds rank 0
   - All processes initialize via NCCL and share model weights
   - Worker processes enter `ModelRunner.loop()` listening on shared memory

2. **Request Processing**:
   - `add_request()`: Tokenizes prompts and creates `Sequence` objects
   - `step()`: Scheduler batches sequences → Model executes → Post-process results
   - `generate()`: High-level API that loops until all sequences complete

3. **Tensor Parallelism**:
   - Rank 0 writes method name + args to shared memory
   - Events signal workers to read and execute
   - All ranks execute the same operation on their data shard
   - NCCL handles all-reduce for collective operations

### Key Design Decisions

- **Multiprocessing over Threading**: Uses `mp.get_context("spawn")` to avoid GIL and ensure proper CUDA context isolation
- **Shared Memory IPC**: 1MB buffer for pickled method calls, synchronized via multiprocessing Events
- **Block-based KV Cache**: Configurable block size (default 256 tokens) for efficient memory management
- **No Greedy Sampling**: Temperature must be > 1e-10 (enforced in `SamplingParams.__post_init__`)
- **Separate Prefill/Decode**: Scheduler distinguishes between prefill (processing prompts) and decode (generating tokens) for batching optimization

## Configuration

Key `Config` parameters in `nanovllm/config.py`:
- `max_num_batched_tokens`: Maximum tokens per batch (default: 16384)
- `max_num_seqs`: Maximum concurrent sequences (default: 512)
- `max_model_len`: Automatically capped by model's `max_position_embeddings`
- `tensor_parallel_size`: Number of GPUs (1-8)
- `enforce_eager`: Disable CUDA graphs (useful for debugging)
- `kvcache_block_size`: Must be multiple of 256

## Model Registration

To add a new model:

1. Create implementation in `nanovllm/models/` (e.g., `new_model.py`)
2. Import in `nanovllm/engine/model_runner.py`
3. Add to `ModelRunner.model_dict`:
   ```python
   model_dict = {
       "qwen3": Qwen3ForCausalLM,
       "new_model": NewModelForCausalLM,  # Add this
   }
   ```

## Important Constraints

- Python 3.10-3.12 only (not 3.13+)
- Requires CUDA-capable GPU
- KV cache block size must be divisible by 256
- Tensor parallel size limited to 8 GPUs maximum
- Temperature < 1e-10 will raise assertion error (use explicit temperature instead of greedy)

## Performance Notes

- CUDA graphs are enabled by default (disable with `enforce_eager=True`)
- Overlap scheduling is mentioned in comments but not implemented
- Prefix caching support mentioned in README but implementation details unclear
- Uses Triton kernels for fused operations in MoE layers
