import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Optional, Tuple, Any
import triton
import triton.language as tl

from ..utils.context import get_context
from .linear import QKVParallelLinear

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
    softmax_scale: float,
    causal: bool = True,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size = cu_seqlens_q.shape[0] - 1
    outputs = []

    # 输入qkv是各个seq沿着len拼接在一起的，需要根据 cu_seqlens_q 分割成单独序列
    for i in range(batch_size):
        # _print_once(f"batchsize: {batch_size} \n")

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
        
        q_i = q_i.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len_q, head_dim]
        k_i = k_i.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, seq_len_k, head_dim]
        v_i = v_i.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, seq_len_k, head_dim]
        
        # GQA: 重复 K 和 V 以匹配 Q 的头数
        num_queries_per_kv = q.shape[1] // k.shape[1]
        if num_queries_per_kv > 1:
            k_i = k_i.repeat_interleave(num_queries_per_kv, dim=1)  # [1, num_heads, seq_len_k, head_dim]
            v_i = v_i.repeat_interleave(num_queries_per_kv, dim=1)  # [1, num_heads, seq_len_k, head_dim]
        
        # 使用 SDPA 计算注意力
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                enable_gqa=True,
                scale=softmax_scale
            )
        
        # 重塑回原始形状: [1, num_heads, seq_len_q, head_dim] -> [seq_len_q, num_heads, head_dim]
        output_i = output_i.transpose(1, 2).squeeze(0)  # [seq_len_q, num_heads, head_dim]
        outputs.append(output_i)
        # _print_once(output_i.shape)
    
    # 拼接所有序列的输出
    return torch.cat(outputs, dim=0)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:

    batch_size = q.shape[0]
    assert q.shape[1] == 1, "Decode stage should have seq_len=1"
    
    q_sdpa = q.transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
    
    # GQA: 重复 K 和 V 以匹配 Q 的头数
    num_queries_per_kv = q.shape[2] // k_cache.shape[2]
    if num_queries_per_kv > 1:
        k_sdpa = k_cache.repeat_interleave(num_queries_per_kv, dim=2)  # [batch_size, cache_seq_len, num_heads, head_dim]
        v_sdpa = v_cache.repeat_interleave(num_queries_per_kv, dim=2)  # [batch_size, cache_seq_len, num_heads, head_dim]
    else:
        k_sdpa = k_cache
        v_sdpa = v_cache
    
    # 转置为 SDPA 格式
    k_sdpa = k_sdpa.transpose(1, 2)  # [batch_size, num_heads, cache_seq_len, head_dim]
    v_sdpa = v_sdpa.transpose(1, 2)  # [batch_size, num_heads, cache_seq_len, head_dim]

    
    # 使用 SDPA 计算注意力
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            enable_gqa=True,
            scale=softmax_scale
        )
    
    output = output.transpose(1, 2)
    return output


def _gather_cache_tokens(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """按照 block_table 的顺序，把 cache 中对应的 token 拼接成连续序列。"""
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

# ==================== Attention 类实现 ====================

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
        
        # GQA support: num_heads must be divisible by num_kv_heads
        assert num_heads % num_kv_heads == 0
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # Initialize KV cache
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播，兼容原有的 flashattention 调用方式
        输入形状: [batch*seq_len, num_heads, head_dim]
        """
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if (
                context.block_tables is not None
                and k_cache.numel()
                and v_cache.numel()
            ):
                block_tables = context.block_tables
                cu_seqlens_k = context.cu_seqlens_k
                gathered_k = []
                gathered_v = []
                for idx in range(block_tables.size(0)):
                    total_len = int(cu_seqlens_k[idx + 1].item() - cu_seqlens_k[idx].item())
                    if total_len == 0:
                        continue
                    blocks = block_tables[idx].to(device="cpu")
                    gathered_k.append(_gather_cache_tokens(k_cache, blocks, total_len))
                    gathered_v.append(_gather_cache_tokens(v_cache, blocks, total_len))
                if gathered_k:
                    k = torch.cat(gathered_k, dim=0)
                    v = torch.cat(gathered_v, dim=0)

            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
            batch_size = context.cu_seqlens_q.shape[0] - 1
            o = o.view(-1, self.num_heads * self.head_dim)
        else:
            # Decode 阶段: 使用 flash_attn_with_kvcache 的兼容实现
            # 重塑 q 为 [batch_size, 1, num_heads, head_dim]
            batch_size = q.shape[0]
            q_reshaped = q.view(batch_size, 1, self.num_heads, self.head_dim)
            
            # 裁剪 KV 缓存以匹配当前 batch_size
            if k_cache.numel() > 0 and v_cache.numel() > 0:
                # 确保 KV 缓存的 batch_size 与当前输入一致
                if k_cache.shape[0] > batch_size:
                    k_cache = k_cache[:batch_size]
                    v_cache = v_cache[:batch_size]
                # 同时裁剪 context_lens 以匹配
                if context.context_lens is not None and context.context_lens.shape[0] > batch_size:
                    context_lens = context.context_lens[:batch_size]
                else:
                    context_lens = context.context_lens
            
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
            
            # 重塑回原始形状
            o = o.view(batch_size, self.num_heads * self.head_dim)
        
        return o
