import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = dist.get_world_size()
        self.num_total_heads = num_heads
        assert self.num_total_heads % tp_size == 0
        self.num_heads = self.num_total_heads // tp_size

        self.num_total_kv_heads = num_kv_heads
        assert self.num_total_kv_heads % tp_size == 0
        self.num_kv_heads = self.num_total_kv_heads // tp_size

        self.head_dim = head_dim if head_dim else self.hidden_size // num_heads
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # QKVParallelLinear内部会处理TP，因此传入num_total_heads
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_total_heads,
            self.num_total_kv_heads,
            self.qkv_bias
        )
        self.o_proj = RowParallelLinear(
            self.num_total_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        print(f"[INFO]: In attention, qkv.shape is {q.shape}, {k.shape}, {v.shape}")
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermeidate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermeidate_size, intermeidate_size],
            False
        )
        self.down_proj = RowParallelLinear(
            intermeidate_size,
            hidden_size,
            bias=False
        )
        assert hidden_act == 'silu'
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        print(f"[INFO]: In MLP, hidden_states.shape is {hidden_states.shape}")
        gate_up = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_act = config.hidden_act
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = ReplicatedLinear(self.hidden_size, self.num_experts, bias=False)
        # 这里每个专家是一个MoeMLP，按照上面的实现，这里是将每个专家的权重分布到多个GPU上，每个GPU持有所有专家的部分权重
        # 这是Sharded Experts的方法
        # 还有一种Split Experts的方法，将专家分布到多个GPU上，每个GPU持有部分专家，在GPU上的专家的权重是完整的
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(self.hidden_size, self.intermediate_size, self.hidden_act) for _ in range(self.num_experts)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        h = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, h)
        router_logit = self.gate(hidden_states)
        # routing_weights.shape = (b * s, num_experts)
        routing_weights = F.softmax(router_logit, dim=-1, dtype=torch.float)
        # selected_experts.shape = (b * s, num_experts_per_tok)
        # After topk, routing_weights.shape = (b * s, num_experts_per_tok)
        # selected_experts[i]代表第i个token要被分配到的专家的索引
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # 找到哪些 token 选择了当前这个专家
            # top_x 代表 token 在 batch 中的索引，top_y 代表这个专家是该 token 的第几个选择
            top_x, top_y = torch.where(selected_experts == expert_idx)
            if top_x.shape[0] == 0:
                continue
            # 提取需要交给该专家处理的 token 数据
            current_state = hidden_states[top_x]
            # 计算专家输出并乘上对应的路由权重
            # current_states经过了expert_layer的分列并行和分行并行，在分行并行层通过all_reduce得到了最终结果current_hidden_states
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, top_y].unsqueeze(-1)
            
            # 将结果加回到对应的位置
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # Reshape 回原始形状
        return final_hidden_states.view(original_shape)

class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermeidate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act
                                )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position: torch.Tensor,
        residual: torch.Tensor | None
    ) -> torch.Tensor:
        # 以下代码中，涉及hidden_states和residual的相加，都在layer_norm中完成，具体可看layer_norm的实现
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, position)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        residual = None
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, positions, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(
        self,
        config: Qwen3MoeConfig
    ):
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids, positions)
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)