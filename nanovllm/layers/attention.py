import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# 放到 import 之后，kernel 定义之前
_print_once_done = False

def _print_once(*args):
    global _print_once_done
    if not _print_once_done:
        print(*args)
        _print_once_done = True


@triton.jit
def store_kvcache_kernel(
    key_ptr,        # 本次生成的 key 的实际物理首地址
    key_stride,     # 告诉 key 如何跳行（dim）
    value_ptr,
    value_stride,
    k_cache_ptr,    # k_cache 的实际物理首地址
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,        # 常量参数，表示 head_dim × num_heads
):
    idx = tl.program_id(0)          # 获取当前 instance 的编号（类似 CUDA 里的 blockIdx.x）。这里每个 program 处理一个 key/value 对应的一行。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)        # 获取偏移量，后续加载内存

    key = tl.load(key_ptr + key_offsets)                        # 根据物理地址进行加载：从新生成的key中读单独一个token的key
    value = tl.load(value_ptr + value_offsets)

    slot = tl.load(slot_mapping_ptr + idx)                  # 计算要写入 kcache 的“行号” slot 值，由 model runner 预先计算好
    cache_offsets = slot * D + tl.arange(0, D)               
    
    tl.store(k_cache_ptr + cache_offsets, key)              # 已知送货地址，已知货物，开始送货
    tl.store(v_cache_ptr + cache_offsets, value)            # 对于 pd 阶段，kernel 只需要获取正确 slotmapping 即可正确写入 kvcache


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
    str = "="*30 + "slot mapping and stride" +"="*30 + '\n'
    _print_once(str, "slot_mapping: ", slot_mapping, "\n key.shape:", key.shape, "\n k_cache.shape:", k_cache.shape,"\n D:", D)
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
            # 计算完 kv 嵌入之后，调用 store_kvcache kernel 存进去当前token的kv
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 根据 PD 阶段，调用不同的 flash attention kernel
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # 通过cu_seqlens_q和cu_seqlens_k来区分不同序列
            # print('='*60)
            # print("slot mapping: \n")
            # print(context.slot_mapping)
            # print('='*60)
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
