import torch
import triton
import triton.language as tl

@triton.jit
def awq_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    # Matrix dimensions
    M, N, K, G,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_sk, stride_sn,
    stride_zk, stride_zn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = A * Dequant(B). 
    A: [M, K] (FP16)
    B: [K, N/8] (Int32)
    C: [M, N] (FP16)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    # --- 1. 计算范围 ---
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # AWQ Shifts: [0, 16, 4, 20, 8, 24, 12, 28]
    idx = tl.arange(0, 8)
    shifts = ((idx % 2) * 4 + (idx // 2)) * 4 # [8]

    # --- 2. 迭代 K 维度 ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A [M, K]
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=(k * BLOCK_SIZE_K + offs_k[None, :]) < K, other=0.0)

        # 加载 B [K, N/8] (打包的 4-bit)
        # 注意：offs_bn 是逻辑列，读取需要除以 8
        offs_bn_packed = (pid_n * BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
        b_ptrs = b_ptr + ((k * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + offs_bn_packed[None, :] * stride_bn)
        b_packed = tl.load(b_ptrs, mask=(k * BLOCK_SIZE_K + offs_k[:, None]) < K, other=0)

        # 加载 Scales & Zeros [K/G, N]
        group_idx = (k * BLOCK_SIZE_K) // G
        s_ptrs = scales_ptr + (group_idx * stride_sk + offs_bn[None, :] * stride_sn)
        s_mask = offs_bn[None, :] < N
        scales = tl.load(s_ptrs, mask=s_mask, other=0.0)

        z_ptrs = zeros_ptr + (group_idx * stride_zk + offs_bn_packed[None, :] * stride_zn)
        z_mask = offs_bn_packed[None, :] < (N // 8)
        zeros_packed = tl.load(z_ptrs, mask=z_mask, other=0) # [1, BLOCK_SIZE_N/8]

        # --- 3. 实时反量化 B ---
        # 解包 B
        b_reshaped = tl.reshape(b_packed, (BLOCK_SIZE_K, BLOCK_SIZE_N // 8, 1))
        b_dequant = (b_reshaped >> shifts[None, None, :]) & 0xF
        b_dequant = tl.reshape(b_dequant, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        
        # 解包 Zeros 并应用
        z_reshaped = tl.reshape(zeros_packed, (1, BLOCK_SIZE_N // 8, 1))
        z_dequant = (z_reshaped >> shifts[None, None, :]) & 0xF
        z_dequant = tl.reshape(z_dequant, (1, BLOCK_SIZE_N))

        # 还原权重: (Q - Z) * S
        b_final = (b_dequant.to(tl.float16) - z_dequant.to(tl.float16)) * scales

        # --- 4. 矩阵乘法累加 ---
        accumulator += tl.dot(a, b_final.to(tl.float16))

    # --- 5. 写回结果 ---
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def awq_matmul(a, qweight, qzeros, scales, group_size):
    """
    Wrapper for C = A * Dequant(B)
    A (Activation): [M, K] - FP16
    qweight: [K, N/8] - Int32 (Packed 4-bit)
    qzeros: [K/G, N/8] - Int32 (Packed 4-bit)
    scales: [K/G, N] - FP16
    """
    # 确保输入是连续的，否则 stride 计算会出错
    a = a.contiguous()
    
    M, K = a.shape
    K_orig, N_packed = qweight.shape
    N = N_packed * 8
    G = group_size

    # 校验维度
    assert K == K_orig, f"Incompatible K dimension: {K} vs {K_orig}"
    
    # 准备输出矩阵
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 定义超参数 (可以根据显卡型号进一步调优)
    # 对于大多数 RTX 卡，128x128 或 64x128 是比较平衡的配置
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8  # 用于 L2 Cache 优化的分组大小

    # 计算 Grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # 启动 Kernel
    awq_matmul_kernel[grid](
        a, qweight, c,
        scales, qzeros,
        M, N, K, G,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return c