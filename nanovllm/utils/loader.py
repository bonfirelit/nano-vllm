import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from nanovllm.config import QuantConfig


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(
    model: nn.Module, 
    path: str, 
    quant_config: QuantConfig | None = None
) -> None:
    # Check if the model has its own load_model method
    if hasattr(model, "load_model"):
        model.load_model(path)
        return
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if 'qzeros' in weight_name or 'scales' in weight_name:
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        if 'qweight' in weight_name:
                            weight_name = weight_name.replace('qweight', 'weight')
                        # 例如，weight_name 是 "layers.0.self_attn.q_proj.weight"
                        # packed_modules_mapping 中 "q_proj" 映射到 ("qkv_proj", "q")
                        # 那么 param_name 就是 "layers.0.self_attn.qkv_proj.weight"
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        if quant_config is not None:
                            prefix = weight_name.split('.')[:-1]
                            qweight = f.get_tensor('.'.join(prefix + ["qweight"]))
                            qzeros = f.get_tensor('.'.join(prefix + ["qzeros"]))
                            scales = f.get_tensor('.'.join(prefix + ["scales"]))
                            weight = _dequantize(qweight, qzeros, scales, quant_config)
                        else:
                            weight = f.get_tensor(weight_name)
                        weight_loader(param, weight, shard_id)
                        break
                else:
                    if 'qweight' in weight_name:
                        weight_name = weight_name.replace('qweight', 'weight')
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    if quant_config is not None and "proj" in weight_name:
                        prefix = weight_name.split('.')[:-1]
                        qweight = f.get_tensor('.'.join(prefix + ["qweight"]))
                        qzeros = f.get_tensor('.'.join(prefix + ["qzeros"]))
                        scales = f.get_tensor('.'.join(prefix + ["scales"]))
                        weight = _dequantize(qweight, qzeros, scales, quant_config)
                    else:
                        weight = f.get_tensor(weight_name)
                    weight_loader(param, weight)


def _dequantize(
    qweight: torch.Tensor, # [K, N // 8] - int32
    qzeros: torch.Tensor, # [K // G, N // 8] - int32
    scales: torch.Tensor, # [K // G, N] - fp16
    config: QuantConfig
) -> torch.Tensor:
    K, N_packed = qweight.shape
    N = N_packed * 8
    G = config.group_size

    # 1. 准备位移向量 (Shifts)
    # AWQ 顺序: [0, 4, 1, 5, 2, 6, 3, 7]
    # 每个索引映射到实际在 int32 中的 4-bit 起始位置 (需乘以 4)
    awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=qweight.device)
    shifts = awq_order * 4  # 结果为 [0, 16, 4, 20, 8, 24, 12, 28]

    # 2. 解包权重 (Unpack qweight)
    # 将 qweight 扩展一维 [K, N_packed, 1] -> [K, N_packed, 8]
    # 然后通过位移提取每个 4-bit 的值
    iweights = qweight.unsqueeze(-1)  
    iweights = (iweights >> shifts) & 0xF
    iweights = iweights.reshape(K, N) # 还原为标准的 [K, N]

    # 3. 解包零点 (Unpack qzeros)
    # 逻辑同权重，zeros 是每组 G 个权重共享一个
    izeros = qzeros.unsqueeze(-1)
    izeros = (izeros >> shifts) & 0xF
    izeros = izeros.reshape(-1, N) # [K // G, N]

    # 4. 广播处理 (Broadcasting)
    # scales 和 izeros 的形状是 [K // G, N]，需要拉伸到与 iweights [K, N] 一致
    # 我们通过 repeat_interleave 在行上复制 G 次
    scales_expanded = scales.repeat_interleave(G, dim=0)
    izeros_expanded = izeros.repeat_interleave(G, dim=0)

    # 5. 执行反量化公式: Weight = (Quantized - Zero) * Scale
    # 注意：某些 AWQ 实现中，计算前需要将 iweights 转为 fp16
    weights = (iweights.to(scales.dtype) - izeros_expanded.to(scales.dtype)) * scales_expanded
    
    return weights.t()