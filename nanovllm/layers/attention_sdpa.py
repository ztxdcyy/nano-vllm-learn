import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Optional, Tuple, Any
import triton
import triton.language as tl

from ..utils.context import get_context

_print_once_done = False

def _print_once(*args):
    global _print_once_done
    if not _print_once_done:
        print(*args)
        _print_once_done = True

# ==================== 保留原有的 store_kvcache kernel ====================
@triton.jit
def store_kvcache_kernel(
    key_ptr,        # 传入kv tensor指针
    key_stride,     # 告诉key value 如何跳行（dim）
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,        # 常量参数，表示 head_dim × num_heads
):
    idx = tl.program_id(0)          # 获取当前 instance 的编号（类似 CUDA 里的 blockIdx.x）。这里每个 program 处理一个 key/value 对应的一行。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)        # 获取偏移量，后续加载内存
    key = tl.load(key_ptr + key_offsets)                        # 从新生成的key中读单独一个token的key
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)                  # 计算存放的slot的值
    cache_offsets = slot * D + tl.arange(0, D)              # 第slot行，从0到D-1
    tl.store(k_cache_ptr + cache_offsets, key)              # 往计算好的slot位置写入新生成的kv
    tl.store(v_cache_ptr + cache_offsets, value)            


def store_kvcache(key: torch.Tensor, 
                value: torch.Tensor, 
                k_cache: torch.Tensor, 
                v_cache: torch.Tensor, 
                slot_mapping: torch.Tensor):
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

# ==================== 构造兼容 flashattention 的 API ====================

def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    causal,
    block_table: Optional[torch.Tensor] = None,         # TODO 实现 prefix caching 的时候，prefill 也需要 kvcache
) -> torch.Tensor:
    batch_size = cu_seqlens_q.shape[0] - 1
    outputs = []

    context = get_context()

    # 输入qkv是各个seq沿着len拼接在一起的，需要根据 cu_seqlens_q 分割成单独序列
    for i in range(batch_size):

        start_q = cu_seqlens_q[i]
        end_q = cu_seqlens_q[i + 1]
        start_k = cu_seqlens_k[i] 
        end_k = cu_seqlens_k[i + 1]
        
        # 获取当前序列的 Q, K, V
        q_i = q[start_q:end_q]      # [seq_len_q, num_heads, head_dim]
        k_i = k[start_k:end_k]      # [seq_len_k, num_kv_heads, head_dim] 
        v_i = v[start_k:end_k]      # [seq_len_k, num_kv_heads, head_dim]
        
        # sdpa need batch dim。重塑为 SDPA 需要的形状: [seq_len, num_heads, head_dim] -> [1, seq_len, num_heads, head_dim]
        seq_len_q = end_q - start_q
        seq_len_k = end_k - start_k

        # unsqueeze 出 bs 维度
        q_i, k_i, v_i = [tensor.unsqueeze(0).transpose(1, 2) for tensor in [q_i, k_i, v_i]]
        
        # 注意不用自己重复 attn head，交给 SDPA 去处理 GQA
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                enable_gqa=True,
                scale=None
            )
        
        # 重塑回原始形状: [1, num_heads, seq_len_q, head_dim] -> [seq_len_q, num_heads, head_dim]
        output_i = output_i.transpose(1, 2).squeeze(0)  # [seq_len_q, num_heads, head_dim]
        outputs.append(output_i)
    
    # 拼接所有序列的输出
    return torch.cat(outputs, dim=0)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:

    batch_size = q.shape[0]
    assert q.shape[1] == 1, "Decode stage should have seq_len=1"
    # CUDAGraph 复用时需要避免 Python 侧取 .item()/.tolist()，所以用纯张量算子收集 KV。
    # block_table 是 [B, num_blocks] 的页表，每个 token 的物理块 = block_table[token // block_size]。
    if block_table is None:
        max_seq_len = int(cache_seqlens.max().item()) if cache_seqlens.numel() else 0
        if max_seq_len <= 0:
            return q.new_zeros(q.shape)
        batch_k = k_cache
        batch_v = v_cache
        if k_cache.size(1) < max_seq_len:
            pad_len = max_seq_len - k_cache.size(1)
            pad_shape = (batch_size, pad_len, k_cache.size(2), k_cache.size(3))
            batch_k = torch.cat([k_cache, k_cache.new_zeros(pad_shape)], dim=1)
            batch_v = torch.cat([v_cache, v_cache.new_zeros(pad_shape)], dim=1)
        kv_valid = torch.arange(max_seq_len, device=cache_seqlens.device).unsqueeze(0) < cache_seqlens.unsqueeze(1)
    else:
        block_size = k_cache.size(1)
        block_table = block_table.to(dtype=torch.long)
        max_blocks = block_table.size(1)
        max_seq_len = max_blocks * block_size
        positions = torch.arange(max_seq_len, device=cache_seqlens.device, dtype=torch.long)
        block_idx = positions // block_size
        offset = positions % block_size
        block_ids = block_table[:, block_idx].clamp(min=0)   # [B, S]
        flat_idx = block_ids * block_size + offset       # [B, S]
        cache_flat_k = k_cache.view(-1, k_cache.size(2), k_cache.size(3))
        cache_flat_v = v_cache.view(-1, v_cache.size(2), v_cache.size(3))
        # 2026年1月12日23:41:33 这里组大buffer很容易OOM
        # 先测试小bsz下能不能跑通，然后在bench_my里做表格对比，找到什么时候OOM
        # 再考虑优化显存，不组大buffer，或者设定utilization，限制实例初始化kvcache-block占用的显存大小。
        batch_k = cache_flat_k[flat_idx]                 # [B, S, num_kv_heads, head_dim]
        batch_v = cache_flat_v[flat_idx]
        kv_valid = positions.unsqueeze(0) < cache_seqlens.unsqueeze(1)   # [B, S] bool
        batch_k = batch_k * kv_valid.unsqueeze(-1).unsqueeze(-1)
        batch_v = batch_v * kv_valid.unsqueeze(-1).unsqueeze(-1)
    attn_mask = kv_valid.unsqueeze(1).unsqueeze(1)                    # [B, 1, 1, S]

    q_sdpa, k_sdpa, v_sdpa = [tensor.transpose(1, 2) for tensor in [q, batch_k, batch_v]]

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            enable_gqa=True,
            scale=None
        )
    
    return output.transpose(1, 2)


def _gather_cache_tokens(
    cache: torch.Tensor,        # 从 cache 中取
    block_ids: torch.Tensor,    # 根据 blockids 取
    total_tokens: int,          # 总共要取多少个 tokens 的 kvcache
) -> torch.Tensor:
    
    block_size = cache.shape[1]
    remaining = total_tokens
    pieces = []

    for block_id in block_ids.tolist():
        if block_id < 0 or remaining <= 0:
            break
        take = min(block_size, remaining)
        pieces.append(cache[block_id, :take])
        remaining -= take
    if remaining > 0:
        raise RuntimeError(
            f"无法从 KV cache 中凑齐 {total_tokens} 个 token，仍缺 {remaining} 个。"
        )
    return torch.cat(pieces, dim=0) if pieces else cache.new_zeros((0, cache.shape[2], cache.shape[3]))


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        
        # Initialize KV cache
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:    # Prefill may also need kvcache for prefix caching
                k, v = k_cache, v_cache

            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                causal=True,
                block_table=context.block_tables
            ) 

        else:   
                # print(f"[DEBUG] context len is: {context.context_lens}")
                # [DEBUG] context len is: tensor([18, 16], device='cuda:0', dtype=torch.int32)
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                causal=False,
            )
        
        return o.view(-1, self.num_heads * self.head_dim)