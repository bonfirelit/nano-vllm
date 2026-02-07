import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class QuantConfig:
    bits: int
    group_size: int
    quant_method: str
    version: str
    zero_point: bool
    modules_to_not_convert: list[str] | None = None

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    quantization_config: QuantConfig | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if 'quantization_config' in self.hf_config:
            self.quantization_config = QuantConfig(
                bits=self.hf_config.quantization_config['bits'],
                group_size=self.hf_config.quantization_config['group_size'],
                quant_method=self.hf_config.quantization_config['quant_method'],
                version=self.hf_config.quantization_config['version'],
                zero_point=self.hf_config.quantization_config['zero_point'],
            )
        if self.quantization_config is not None:
            assert self.quantization_config.quant_method in ["awq"], f"Only supported awq"
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
