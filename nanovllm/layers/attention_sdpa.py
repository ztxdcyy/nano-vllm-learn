import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from torch.nn.attention import SDPBackend
from typing import Optional, Tuple, Any
import triton
import triton.language as tl

from ..utils.context import get_context
from .linear import QKVParallelLinear

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
    tl.store(v_cache_ptr + cache_offsets, value)            # 那我有个问题，对于prefill和decode阶段，计算的kvcache长度是不一样的呀？具体还得看flash_attn是怎么计算的，输入输出啥样的，如何处理pd阶段


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
    # 启动triton kernel：one dimension grid
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


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
        context = get_context()
        
        # Input shapes from qwen3.py are already:
        # Q: [batch*seq_len, num_heads, head_dim]
        # K: [batch*seq_len, num_kv_heads, head_dim] 
        # V: [batch*seq_len, num_kv_heads, head_dim]
        
        # For SDPA, we need to reshape to [batch, seq_len, num_heads, head_dim]
        batch_size = q.shape[0] // (context.max_seqlen_q if context.is_prefill else 1)
        seq_len = context.max_seqlen_q if context.is_prefill else 1
        
        # Reshape Q: [batch*seq_len, num_heads, head_dim] -> [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape K, V: [batch*seq_len, num_kv_heads, head_dim] -> [batch, seq_len, num_kv_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Store KV cache if needed
        if context.slot_mapping is not None and hasattr(self, 'k_cache') and hasattr(self, 'v_cache'):
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        
        # Handle different stages: Prefill vs Decode
        if context.is_prefill:
            # Prefill stage: full sequence attention
            output = self._prefill_attention(q, k, v, context)
        else:
            # Decode stage: incremental attention with KV cache
            output = self._decode_attention(q, k, v, context)
        
        return output
    
    def _prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        """Prefill stage attention with full sequence."""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Reshape for SDPA: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # For GQA, repeat K and V to match Q heads
        # K, V shapes: [batch, seq_len, num_kv_heads, head_dim]
        k = k.repeat_interleave(self.num_queries_per_kv, dim=2)  # [batch, seq_len, num_heads, head_dim]
        v = v.repeat_interleave(self.num_queries_per_kv, dim=2)  # [batch, seq_len, num_heads, head_dim]
        
        # Reshape for SDPA: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Use SDPA with math backend - shapes match SDPA requirements:
        # Q: [batch, num_heads, seq_len, head_dim] -> (N, Hq, L, E)
        # K: [batch, num_heads, seq_len, head_dim] -> (N, H, S, E)  
        # V: [batch, num_heads, seq_len, head_dim] -> (N, H, S, Ev)
        with sdp_kernel(backends=[SDPBackend.MATH]):
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=True
            )
        
        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        
        # Reshape to match qwen3.py expected output: [batch, seq_len, num_heads, head_dim] -> [batch*seq_len, num_heads * head_dim]
        output = output.reshape(-1, self.num_heads * self.head_dim)
        return output
    
    def _decode_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        """Decode stage attention with KV cache."""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # For decode, we typically have seq_len=1
        assert seq_len == 1
        
        # Reshape Q for SDPA: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        
        # In decode stage, we need to use the full KV cache from previous tokens
        # The current K, V are only for storage, not for attention computation
        # For actual attention computation, we need to retrieve the cached K, V
        # using context.block_tables and context.context_lens
        
        # For now, as a placeholder implementation, we'll use the cached K, V
        # In a complete implementation, you would:
        # 1. Retrieve cached K, V using block_tables and context_lens
        # 2. Concatenate with current K, V if needed
        # 3. Use the full sequence for attention
        
        # Use cached K, V for attention computation
        if hasattr(self, 'k_cache') and hasattr(self, 'v_cache') and self.k_cache.numel() and self.v_cache.numel():
            # For GQA, repeat cached K and V to match Q heads
            k_attn = self.k_cache.repeat_interleave(self.num_queries_per_kv, dim=2)  # [batch, context_lens, num_heads, head_dim]
            v_attn = self.v_cache.repeat_interleave(self.num_queries_per_kv, dim=2)  # [batch, context_lens, num_heads, head_dim]
            
            # Reshape for SDPA: [batch, context_lens, num_heads, head_dim] -> [batch, num_heads, context_lens, head_dim]
            k_attn = k_attn.transpose(1, 2)  # [batch, num_heads, context_lens, head_dim]
            v_attn = v_attn.transpose(1, 2)  # [batch, num_heads, context_lens, head_dim]
            
            # Use SDPA with math backend - shapes match SDPA requirements:
            # Q: [batch, num_heads, 1, head_dim] -> (N, Hq, L, E)
            # K: [batch, num_heads, context_lens, head_dim] -> (N, H, S, E)
            # V: [batch, num_heads, context_lens, head_dim] -> (N, H, S, Ev)
            with sdp_kernel(backends=[SDPBackend.MATH]):
                output = F.scaled_dot_product_attention(
                    q, k_attn, v_attn,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=True
                )
        else:
            # Fallback to current K, V if cache is not available
            # For GQA, repeat K and V to match Q heads
            k_attn = k.repeat_interleave(self.num_queries_per_kv, dim=2)  # [batch, 1, num_heads, head_dim]
            v_attn = v.repeat_interleave(self.num_queries_per_kv, dim=2)  # [batch, 1, num_heads, head_dim]
            
            # Reshape for SDPA: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
            k_attn = k_attn.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
            v_attn = v_attn.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
            
            # Use SDPA with math backend
            with sdp_kernel(backends=[SDPBackend.MATH]):
                output = F.scaled_dot_product_attention(
                    q, k_attn, v_attn,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=True
                )
        
        # Reshape back: [batch, num_heads, 1, head_dim] -> [batch, 1, num_heads, head_dim]
        output = output.transpose(1, 2)  # [batch, 1, num_heads, head_dim]
        
        # Reshape to match qwen3.py expected output: [batch, 1, num_heads, head_dim] -> [batch, num_heads * head_dim]
        output = output.reshape(-1, self.num_heads * self.head_dim)
        return output