import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


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

    # slot_mapping_ptr 是开始kvcache的物理地址，idx 是力工的编号，slot 是这一小队力工每个人对应一块砖头，总共对应一连串的砖头要搬运
    slot = tl.load(slot_mapping_ptr + idx)                  
    cache_offsets = slot * D + tl.arange(0, D)              # 第slot行，从0到D-1
    
    tl.store(k_cache_ptr + cache_offsets, key)              # 往计算好的slot位置写入新生成的kv
    tl.store(v_cache_ptr + cache_offsets, value)            # 那我有个问题，对于prefill和decode阶段，计算的kvcache长度是不一样的呀？具体还得看flash_attn是怎么计算的，输入输出啥样的，如何处理pd阶段


def store_kvcache(key: torch.Tensor, 
                value: torch.Tensor, 
                k_cache: torch.Tensor, 
                v_cache: torch.Tensor, 
                slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动triton kernel：one dimension grid
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale          # softmax_scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor

        # flatten batch size dim for flash attention
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 根据 PD 阶段，调用不同的 flash attention kernel
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # 通过cu_seqlens_q和cu_seqlens_k来区分不同序列
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # 通过context.context_lens来区分不同序列
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
