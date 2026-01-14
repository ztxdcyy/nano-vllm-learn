import triton
import triton.language as tl
import torch
import torch.nn as nn

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    将新生成的 KV 写入分页 KV cache。
    - grid[0] 对应 token 维度；grid[1] 对应 kv head。
    - slot_mapping 提供逻辑 token 在全局 KV cache 的物理地址。
    """
    token_idx = tl.program_id(0)  # 当前处理的 token
    head_idx = tl.program_id(1)   # 当前处理的 kv head

    # 计算 token 对应的物理 slot（-1 表示无效）
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    # 将全局 slot 映射到 block/table 位置
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # head 内部的连续偏移 [0, 1, ..., head_dim-1]
    head_offsets = tl.arange(0, head_dim)

    # 输入 key/value 是 (num_tokens, num_kv_heads, head_dim)
    input_offset = (
        token_idx * num_kv_heads * head_dim
        + head_idx * head_dim
        + head_offsets
    )

    # KV cache 布局: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (
        block_idx * block_size * num_kv_heads * head_dim
        + block_offset * num_kv_heads * head_dim
        + head_idx * head_dim
        + head_offsets
    )

    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)

    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    """
    将一批 KV 写入分页缓存。

    Args:
        key/value: (num_tokens, num_kv_heads, head_dim)
        k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) 每个 token 的物理写入位置
        block_size: 每个 block 的 token 数，需与 cache 分配一致
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    assert k_cache.shape == v_cache.shape
    assert slot_mapping.numel() == num_tokens

    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    grid = (num_tokens, num_kv_heads)
    store_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    变长序列的 FlashAttention 内核。
    每个 program 处理一个序列、一个 head、一个 query block。
    """
    start_m = tl.program_id(0)  # query block idx
    off_h = tl.program_id(1)    # head idx
    seq_idx = tl.program_id(2)  # sequence idx

    kv_head_idx = off_h // (num_heads // num_kv_heads)  # GQA 映射

    # 读取当前序列的 q 范围
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    if start_m * BLOCK_M >= seq_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Q shape: (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 在线 softmax 累积
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1e10, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        # K shape: (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        qk = tl.dot(q, k) * scale

        # 因果 mask：只能看到当前位置及之前
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        # 在线 softmax（保持数值稳定）
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = acc * alpha[:, None]

        # V shape: (total_tokens, num_kv_heads, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        acc = acc + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    acc = acc / l_i[:, None]

    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """变长 prefill 阶段的 Triton FlashAttention。"""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    output = torch.empty_like(q)

    # 共享内存估算：BLOCK_M * BLOCK_N * 4 bytes (score) + Q/K/V tile
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    num_seqs = cu_seqlens.shape[0] - 1
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    flash_attention_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return output


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    分页 KV cache 的 decode 内核。
    - grid[0]: batch 维度
    - grid[1]: head 维度
    每个 program 遍历当前序列的 block_table，按块在线 softmax 累加。
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    context_len = tl.load(context_lens_ptr + batch_idx)

    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    # 逐 chunk 读取 block_table 并聚合
    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N
        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len

            # 预先填充 -inf，后续再替换有效位置
            qk = tl.full([BLOCK_N], -1e10, dtype=tl.float32)

            # chunk内，计算 QK^T
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        # 计算物理块索引
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        if physical_block_idx != -1:
                            k_offset = (
                                physical_block_idx * block_size * num_kv_heads * head_dim
                                + block_offset * num_kv_heads * head_dim
                                + kv_head_idx * head_dim
                                + offs_d
                            )
                            k_vec = tl.load(k_cache_ptr + k_offset)
                            score = tl.sum(q * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)

            qk = tl.where(mask_n, qk, -1e10)

            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            acc = acc * alpha
            l_i = l_i * alpha

            # 累加 V
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        if physical_block_idx != -1:
                            v_offset = (
                                physical_block_idx * block_size * num_kv_heads * head_dim
                                + block_offset * num_kv_heads * head_dim
                                + kv_head_idx * head_dim
                                + offs_d
                            )
                            v_vec = tl.load(v_cache_ptr + v_offset)
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            acc = acc + weight * v_vec
                            l_i = l_i + weight

            m_i = m_i_new

    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """分页 KV cache 的 decode 前向。"""
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    query = query.contiguous()
    output = torch.empty_like(query)

    BLOCK_N = 64 if head_dim <= 128 else 32
    grid = (batch_size, num_heads)

    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    return output


class Attention(nn.Module):
    """
    Triton 版注意力，prefill 用 varlen FlashAttention，decode 用分页 KV。
    注意：block_size 需与 ModelRunner 分配 KV cache 的 block_size 一致。
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        block_size: int = 256,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 将新 KV 写入分页 cache
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            if k.dim() == 4:
                # (B, N, num_kv_heads, head_dim) -> (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # (num_tokens, num_kv_heads * head_dim) -> (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
                v_to_store = v.view(-1, self.num_kv_heads, self.head_dim).contiguous()

            store_kvcache(
                k_to_store,
                v_to_store,
                k_cache,
                v_cache,
                context.slot_mapping,
                self.block_size,
            )

        # softmax scale
        scale = self.scale / (self.head_dim**0.5)

        if context.is_prefill:
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            o = flash_attention_prefill(
                q,
                k,
                v,
                cu_seqlens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)

        o = paged_attention_decode(
            q,
            k_cache,
            v_cache,
            context.block_tables,
            context.context_lens,
            scale,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )
        return o.reshape(o.shape[0], self.num_heads * self.head_dim)
