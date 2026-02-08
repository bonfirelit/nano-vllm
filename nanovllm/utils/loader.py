import os
from glob import glob
import torch
import triton
import triton.language as tl
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
                            qweight = f.get_tensor('.'.join(prefix + ["qweight"])).cuda()
                            qzeros = f.get_tensor('.'.join(prefix + ["qzeros"])).cuda()
                            scales = f.get_tensor('.'.join(prefix + ["scales"])).cuda()
                            # weight = _dequantize(qweight, qzeros, scales, quant_config)
                            weight = triton_dequantize(qweight, scales, qzeros, quant_config)
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
                        qweight = f.get_tensor('.'.join(prefix + ["qweight"])).cuda()
                        qzeros = f.get_tensor('.'.join(prefix + ["qzeros"])).cuda()
                        scales = f.get_tensor('.'.join(prefix + ["scales"])).cuda()
                        weight = triton_dequantize(qweight, scales, qzeros, quant_config)
                        # weight = _dequantize(qweight, qzeros, scales, quant_config)
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
@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,      # 指针: [K, N/8]
    scales_ptr,       # 指针: [K/G, N]
    zeros_ptr,        # 指针: [K/G, N/8]
    result_ptr,       # 指针: [K, N] (输出)
    K, N, G,          # 维度
    stride_qk, stride_qn,  # qweight 的步长
    stride_sk, stride_sn,  # scales 的步长
    stride_zk, stride_zn,  # zeros 的步长
    stride_rk, stride_rn,  # result 的步长
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 1. 索引计算
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n_logical = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 针对打包数据的 N 索引 (每 8 个逻辑列对应一个打包列)
    # 这里的关键是：每一个线程块处理的打包列范围
    offs_n_packed = (pid_n * BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)

    # 2. 构造 AWQ 位移 (0, 16, 4, 20, 8, 24, 12, 28)
    # 对应顺序 [0, 4, 1, 5, 2, 6, 3, 7]
    idx = tl.arange(0, 8)
    shifts = ((idx % 2) * 4 + (idx // 2)) * 4

    # 3. 加载 qweight [BLOCK_SIZE_K, BLOCK_SIZE_N / 8]
    q_ptrs = qweight_ptr + (offs_k[:, None] * stride_qk + offs_n_packed[None, :] * stride_qn)
    q_mask = (offs_k[:, None] < K) & (offs_n_packed[None, :] < (N // 8))
    q_packed = tl.load(q_ptrs, mask=q_mask, other=0)

    # 4. 解包逻辑：利用广播和 reshape
    # 将 q_packed 展开一个新维度：[K, N/8] -> [K, N/8, 1]
    q_res = tl.reshape(q_packed, (BLOCK_SIZE_K, BLOCK_SIZE_N // 8, 1))
    # 广播 shifts: [1, 1, 8]
    shifts_br = tl.reshape(shifts, (1, 1, 8))
    
    # 利用广播机制直接计算
    weights = (q_res >> shifts_br) & 0xF
    # 展平回 [K, N]
    weights = tl.reshape(weights, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # 5. 加载 Zeros (同理)
    group_idx = offs_k // G
    z_ptrs = zeros_ptr + (group_idx[:, None] * stride_zk + offs_n_packed[None, :] * stride_zn)
    z_mask = (group_idx[:, None] < (K // G)) & (offs_n_packed[None, :] < (N // 8))
    z_packed = tl.load(z_ptrs, mask=z_mask, other=0)
    
    z_res = tl.reshape(z_packed, (BLOCK_SIZE_K, BLOCK_SIZE_N // 8, 1))
    zeros = (z_res >> shifts_br) & 0xF
    zeros = tl.reshape(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))
    
    # 6. 加载 Scales (Scales 是正常的 FP16，无需位移)
    s_ptrs = scales_ptr + (group_idx[:, None] * stride_sk + offs_n_logical[None, :] * stride_sn)
    s_mask = (group_idx[:, None] < (K // G)) & (offs_n_logical[None, :] < N)
    scales = tl.load(s_ptrs, mask=s_mask, other=0.0)
    
    # 7. 反量化计算
    # Weight = (Q - Z) * S
    dequant_weights = (weights.to(tl.float16) - zeros.to(tl.float16)) * scales
    
    # 8. 写回结果
    r_ptrs = result_ptr + (offs_k[:, None] * stride_rk + offs_n_logical[None, :] * stride_rn)
    r_mask = (offs_k[:, None] < K) & (offs_n_logical[None, :] < N)
    tl.store(r_ptrs, dequant_weights, mask=r_mask)

def triton_dequantize(qweight, scales, qzeros, config):
    K, N_packed = qweight.shape
    N = N_packed * 8
    G = config.group_size
    
    result = torch.empty((K, N), dtype=scales.dtype, device=qweight.device)
    
    # 定义 Grid：按输出矩阵的 K 和 N 划分
    # BLOCK_SIZE_N 必须是 8 的倍数（因为 1 个 int32 包含 8 个权重）
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    awq_dequantize_kernel[grid](
        qweight, scales, qzeros, result,
        K, N, G,
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        result.stride(0), result.stride(1),
        BLOCK_SIZE_K=32, 
        BLOCK_SIZE_N=128, # 一个块处理 128 列（对应 16 个 int32）
    )
    
    return result.t()