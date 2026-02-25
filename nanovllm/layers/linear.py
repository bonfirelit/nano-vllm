import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from nanovllm.utils.loader import QuantConfig
from nanovllm.layers.quantization.awq_matmul import awq_matmul

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: QuantConfig | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        if quant_config is not None:
            self.group_size = quant_config.group_size
            self.qweight = nn.Parameter(torch.empty(
                input_size,
                output_size // 8,
                dtype=torch.int32),
                requires_grad=False,
            )
            self.qzeros = nn.Parameter(torch.empty(
                input_size // self.group_size,
                output_size // 8,
                dtype=torch.int32),
                requires_grad=False,
            )
            self.scales = nn.Parameter(torch.empty(
                input_size // self.group_size,
                output_size,
                dtype=torch.float16),
                requires_grad=False,
            )
            self.qweight.weight_loader = self.weight_loader
            self.qzeros.weight_loader = self.weight_loader
            self.scales.weight_loader = self.weight_loader
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            # self.weight.name = "weight"
            # setattr(self.weight, "name", "weight")
            self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            # self.bias.name = "bias"
            # setattr(self.bias, "name", "bias")
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ReplicatedLinear表示虽然是分布式的环境，但每个GPU都持有一样的权重，并未分割
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
    ):
        super().__init__(input_size, output_size, bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if param.data.shape != loaded_weight.shape:
            raise ValueError(f"Shape mismatch: {param.data.shape} vs {loaded_weight.shape}")
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "qweight"):
            output = awq_matmul(x, self.qweight, self.qzeros, self.scales, self.group_size)
            if self.bias is not None:
                output += self.bias
            return output
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        tp_rank = self.tp_rank
        
        # 核心逻辑：获取当前参数在哪个维度需要被切分
        # 如果是 qweight 或 qzeros，维度 1 对应 output_size
        # 如果是 scales 或 bias 或普通的 weight，维度 0/1 对应关系不同
        
        # 1. 确定切分维度
        # 对于 AWQ，qweight 是 [in, out/8]，qzeros 是 [in/g, out/8]
        # 它们的 output 维度都在 dim 1
        if hasattr(self, "qweight") and param is self.qweight:
            curr_tp_dim = 1 
        elif hasattr(self, "scales") and param is self.scales:
            curr_tp_dim = 1
        elif hasattr(self, "bias") and param is self.bias:
            curr_tp_dim = 0
        else:
            # 普通 FP16 weight 是 [out, in]，所以切分 dim 0
            curr_tp_dim = 0
        # 2. 计算切片并加载
        shard_size = param.shape[curr_tp_dim]
        start_idx = tp_rank * shard_size
        
        loaded_weight = loaded_weight.narrow(curr_tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "qweight") and self.qweight is not None:
            # 这里的 output 已经是当前 rank 的分片结果 [batch, output_size/tp_size]
            output = awq_matmul(x, self.qweight, self.qzeros, self.scales, self.group_size)
            if self.bias is not None:
                output += self.bias
            return output
        
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: QuantConfig | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        if hasattr(self, "qweight") and param is self.qweight:
            param_name = "qweight"
        elif hasattr(self, "qzeros") and param is self.qzeros:
            param_name = "qzeros"
        elif hasattr(self, "scales") and param is self.scales:
            param_name = "scales"
        elif hasattr(self, "bias") and param is self.bias:
            param_name = "bias"
        else:
            param_name = "weight"
        print(f"MergedColumn DEBUG: param={param_name}, shape={param.shape}, "
              f"shard={loaded_shard_id}, loaded_shape={loaded_weight.shape}")
        # 1. 确定当前参数在 output 维度上的缩放系数
        # qweight/qzeros 是打包过的 (int32)，每个元素代表 8 个权重
        # scales/bias 是 1:1 的
        pack_factor = 8 if (param_name == "qweight" or param_name == "qzeros") else 1
        # 2. 确定切分维度 (对于 ColumnParallel，output 维即我们要切的维)
        # qweight/qzeros/scales: 维度 1
        # bias: 维度 0
        if param_name in ["bias", "weight"]:
            tp_dim = 0
        else:
            tp_dim = 1
        # 3. 计算在目标 Parameter (已合并) 中的偏移量
        # 这里的 offset 和 size 都需要考虑 pack_factor
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // (self.tp_size * pack_factor)
        shard_size = self.output_sizes[loaded_shard_id] // (self.tp_size * pack_factor)
        # 4. 定位到当前 Parameter 内部对应的“槽位”
        target_slice = param_data.narrow(tp_dim, shard_offset, shard_size)
        # 5. 对加载进来的原始权重 (尚未切分 TP) 进行切片
        # 注意：loaded_weight 是 [in, out_i/pack] 或 [out_i/pack]
        loaded_shard_size = loaded_weight.size(tp_dim) // self.tp_size
        start_idx = self.tp_rank * loaded_shard_size
        loaded_weight_shard = loaded_weight.narrow(tp_dim, start_idx, loaded_shard_size)
        # 6. 拷贝数据
        target_slice.copy_(loaded_weight_shard)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        if hasattr(self, "qweight") and param is self.qweight:
            param_name = "qweight"
        elif hasattr(self, "qzeros") and param is self.qzeros:
            param_name = "qzeros"
        elif hasattr(self, "scales") and param is self.scales:
            param_name = "scales"
        elif hasattr(self, "bias") and param is self.bias:
            param_name = "bias"
        else:
            param_name = "weight"
        print(f"QKVParallel DEBUG: param={param_name}, shape={param.shape}, "
              f"shard={loaded_shard_id}, loaded_shape={loaded_weight.shape}")
        # 1. 确定打包因子和切分维度
        pack_factor = 8 if (param_name == "qweight" or param_name == "qzeros") else 1
        # ColumnParallel 逻辑：除了 bias 在 dim 0，其余（qweight, qzeros, scales）都在 dim 1
        tp_dim = 0 if param_name in ["bias", "weight"] else 1
        # 2. 计算每个部分在当前 rank 内存中的原始宽度 (未打包宽度)
        q_width = self.num_heads * self.head_size
        k_width = self.num_kv_heads * self.head_size
        v_width = self.num_kv_heads * self.head_size
        # 3. 根据加载的 shard 类型 (q, k, v)，计算在合并参数中的偏移和大小
        # 所有的偏移和大小都要除以 pack_factor
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = q_width // pack_factor
        elif loaded_shard_id == "k":
            shard_offset = q_width // pack_factor
            shard_size = k_width // pack_factor
        elif loaded_shard_id == "v":
            shard_offset = (q_width + k_width) // pack_factor
            shard_size = v_width // pack_factor
        else:
            raise ValueError(f"Unknown shard id: {loaded_shard_id}")

        # 4. 定位本地 Parameter 的切片 (Destination)
        target_slice = param_data.narrow(tp_dim, shard_offset, shard_size)

        # 5. 处理从文件读入的加载权重 (Source)
        # 磁盘上的 loaded_weight 形状通常是 [in, full_out_i/pack]
        # 我们需要按 TP 挖出当前 rank 负责的那一块
        # 注意：磁盘上的数据也是打包过的，所以要按当前物理维度计算精确分片
        loaded_shard_size = loaded_weight.size(tp_dim) // self.tp_size
        start_idx = self.tp_rank * loaded_shard_size
        loaded_weight_tp_shard = loaded_weight.narrow(tp_dim, start_idx, loaded_shard_size)

        # 6. 执行拷贝
        target_slice.copy_(loaded_weight_tp_shard)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if hasattr(self, "qweight") and param is self.qweight:
            param_name = "qweight"
        elif hasattr(self, "qzeros") and param is self.qzeros:
            param_name = "qzeros"
        elif hasattr(self, "scales") and param is self.scales:
            param_name = "scales"
        elif hasattr(self, "bias") and param is self.bias:
            param_name = "bias"
        else:
            param_name = "weight"
        print(f"RowParallel DEBUG: param={param_name}, shape={param.shape}, "
              f"loaded_shape={loaded_weight.shape}")
        tp_rank = self.tp_rank
        
        # 1. 确定切分维度
        # 在 RowParallel 中，我们要切分的是 input 维度
        # qweight: [input, output/8] -> 切 dim 0
        # qzeros: [input/g, output/8] -> 切 dim 0
        # scales: [input/g, output] -> 切 dim 0
        # bias: [output] -> 不切分 (只有 Rank 0 或全量持有)
        
        if param_name == "bias":
            # Bias 在 RowParallel 中通常不切分，直接全量拷贝
            param.data.copy_(loaded_weight)
            return

        # 2. 计算切分
        # 对于 qweight，shard_size 就是 input_size // tp_size
        # 对于 qzeros/scales，由于是按 group 存的，shard_size 自动变为 (input/g) // tp_size
        shard_size = param.shape[0] 
        start_idx = tp_rank * shard_size
        
        loaded_weight_shard = loaded_weight.narrow(0, start_idx, shard_size)
        param.data.copy_(loaded_weight_shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "qweight") and self.qweight is not None:
            # 每个 rank 计算局部乘积
            # x: [batch, input_size/tp_size], qweight: [input_size/tp_size, output_size/8]
            output = awq_matmul(x, self.qweight, self.qzeros, self.scales, self.group_size)
        else:
            output = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        # RowParallel 的灵魂：All-Reduce
        if self.tp_size > 1:
            dist.all_reduce(output)

        # 在 all-reduce 之后再加 bias (AWQ 情况下所有 rank 都有完整 bias)
        if hasattr(self, "qweight") and self.qweight is not None and self.bias is not None:
            output += self.bias

        return output
        # 每个GPU上都会执行y = F.linear()，但每个GPU上的y都是最终结果的一部分
        # y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # if self.tp_size > 1:
        #     # 这里进行all_reduce操作，使得每个GPU都得到最终结果
        #     dist.all_reduce(y)
        # return y
