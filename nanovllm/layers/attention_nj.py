from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanovllm.utils.context import get_context


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    # 安全性断言
    # [B, num_kv_heads, head_dim] = [28, 8, 128]
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # N: 当前 step 所有 token 数（可能是 batch 拼接）
    # slot_mapping shape: [N], 表示这 N 个 token 对应应该写入 kv_cache 的哪些位置（token 级别），slot_mapping 在 prepare_prefill 已经构建完毕，其已经绑定了相应的写入 kv cache 的位置，不会覆盖其它请求的 kv cache
    # k_cache, v_cache shape: [num_blocks, block_size, num_heads, head_dim] 
    # 全体 KV 缓存，按 block 组织；总共可容纳 num_blocks * block_size 个 token

    # 需要将 key 和 value 的内容，按照 slot_mapping 中记录的位置写入到 k_cache or v_cache 中
    # 将 key 和 value reshape 为 [N, D]
    # key: [N, num_heads, head_dim] -> [N, D]
    key = key.view(N, D)
    value = value.view(N, D)

    # 将 kv_cache reshape 为 [T, D]，T = total_slots = num_blocks * block_size
    # k_cache : [num_blocks, block_size, num_heads, head_dim] -> [T, D]
    k_cache = k_cache.view(-1, D)
    v_cache = v_cache.view(-1, D)

    # 使用 slot_mapping 将 key/value 写入对应位置
    # slot_mapping: [N]，每个元素都是 kv_cache 中 [0, T) 的位置索引
    # slot_mapping: range(0, 17)+range(256, 270)
    k_cache[slot_mapping] = key
    v_cache[slot_mapping] = value


# 这个函数好像没用？
# def _gather_kv_by_blocktable(
#     k_cache: torch.Tensor,  # [num_blocks, block_size, H, D]
#     v_cache: torch.Tensor,  # [num_blocks, block_size, H, D]
#     block_table_row: torch.Tensor,  # [num_used_blocks] or [max_blocks]
#     seqlen: int,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     按 block_table 把前 seqlen 个 token 的 K/V 拉平成连续张量:
#       返回 k, v 形状: [seqlen, H, D]
#     """
#     num_blocks, block_size, H, D = k_cache.shape
#     device = k_cache.device
#     # 逐 token 反解：(block_id, offset)
#     t = torch.arange(seqlen, device=device)
#     blk_idx = t // block_size
#     off_idx = t % block_size
#     # 取出实际 block_id
#     block_ids = block_table_row[blk_idx]         # [seqlen]
#     # 索引收集
#     k = k_cache[block_ids, off_idx]              # [seqlen, H, D]
#     v = v_cache[block_ids, off_idx]              # [seqlen, H, D]
#     return k, v


def _align_kv_to_q_heads(k: torch.Tensor, v: torch.Tensor, Hq: int):
    """
    k, v: [..., Hk, D]
    将 K/V 在 head 维对齐到 Hq（用于 GQA），要求 Hq 是 Hk 的整数倍。
    返回: [..., Hq, D]
    """
    Hk = k.size(-2)
    assert Hq % Hk == 0, f"GQA head mismatch: Hq={Hq}, Hk={Hk}"
    g = Hq // Hk
    if g == 1:
        return k, v
    # 在 head 维（dim=-2） repeat g 次
    k = k.repeat_interleave(g, dim=-2)
    v = v.repeat_interleave(g, dim=-2)
    return k, v


@torch.no_grad()
def _prefill_causal_mask(Sq: int, Sk: int, device) -> torch.Tensor:
    offset = Sk - Sq
    q_idx = torch.arange(Sq, device=device)
    k_idx = torch.arange(Sk, device=device)
    max_k = offset + q_idx
    return (k_idx.unsqueeze(0) > max_k.unsqueeze(1))  # [Sq, Sk], True=mask


def pytorch_attn_varlen_func(
    q: torch.Tensor,               # [total_q, Hq, D]
    k: torch.Tensor,               # [total_k, Hk, D]  或  [NB, BS, Hk, D]
    v: torch.Tensor,               # 同上
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,    # [B+1]
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,    # [B+1]
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    block_table: Optional[torch.Tensor] = None
) -> torch.Tensor:
    assert q.dim() == 3, "q must be [total_q, Hq, D]"
    B = cu_seqlens_q.numel() - 1
    Hq = q.size(1)
    D = q.size(2)
    device = q.device
    dtype = q.dtype
    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    use_paged = (k.dim() == 4 and v.dim() == 4 and block_table is not None)
    outs = []

    for b in range(B):
        q0, q1 = cu_seqlens_q[b].item(), cu_seqlens_q[b+1].item()
        k0, k1 = cu_seqlens_k[b].item(), cu_seqlens_k[b+1].item()
        Sq, Sk = q1 - q0, k1 - k0
        if Sq == 0:
            continue

        q_b = q[q0:q1]  # [Sq, Hq, D]

        # 得到该样本的 [Sk, Hk, D]
        if use_paged:
            # 按 block_table 收集（与你前面的假设一致）
            num_blocks, block_size = k.size(0), k.size(1)
            t = torch.arange(Sk, device=device)
            blk = t // block_size
            off = t % block_size
            block_ids = block_table[b][blk]        # [Sk]
            k_b = k[block_ids, off]                # [Sk, Hk, D]
            v_b = v[block_ids, off]                # [Sk, Hk, D]
        else:
            k_b = k[k0:k1]                          # [Sk, Hk, D]
            v_b = v[k0:k1]                          # [Sk, Hk, D]

        # === GQA 头数对齐：将 Hk 对齐到 Hq ===
        Hk = k_b.size(1)
        k_b, v_b = _align_kv_to_q_heads(k_b, v_b, Hq)  # -> [Sk, Hq, D]

        # 分数: [Hq, Sq, Sk]
        scores = torch.einsum("qhd,khd->hqk", q_b, k_b).to(torch.float32)
        scores = scores * softmax_scale
        if causal:
            mask = _prefill_causal_mask(Sq, Sk, device=device)  # [Sq, Sk]
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1).to(dtype)              # [Hq, Sq, Sk]
        out_b = torch.einsum("hqk,khd->qhd", attn, v_b)         # [Sq, Hq, D]
        outs.append(out_b)

    return torch.cat(outs, dim=0) if outs else torch.empty_like(q)


def pytorch_attn_with_kvcache(
    q: torch.Tensor,               # [B, 1, Hq, D]
    k_cache: torch.Tensor,         # [T, Hk, D]  或  [NB, BS, Hk, D]
    v_cache: torch.Tensor,         # 同上
    *,
    cache_seqlens: torch.Tensor,   # [B]
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    assert q.dim() == 4 and q.size(1) == 1, "q must be [B, 1, Hq, D]"
    B, _, Hq, D = q.shape
    device, dtype = q.device, q.dtype
    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    use_paged = (k_cache.dim() == 4 and v_cache.dim()
                 == 4 and block_table is not None)
    outs = []

    for b in range(B):
        seqlen = int(cache_seqlens[b].item())
        q_b = q[b, 0]  # [Hq, D]
        if seqlen == 0:
            outs.append(torch.zeros(Hq, D, device=device, dtype=dtype))
            continue

        # 得到该样本的 [seqlen, Hk, D]
        if use_paged:
            NB, BS = k_cache.size(0), k_cache.size(1)
            t = torch.arange(seqlen, device=device)
            blk = t // BS
            off = t % BS
            block_ids = block_table[b][blk]
            k_b = k_cache[block_ids, off]         # [seqlen, Hk, D]
            v_b = v_cache[block_ids, off]
        else:
            k_b = k_cache[:seqlen]                 # [seqlen, Hk, D]
            v_b = v_cache[:seqlen]

        # === GQA 头数对齐：将 Hk 对齐到 Hq ===
        Hk = k_b.size(1)
        k_b, v_b = _align_kv_to_q_heads(k_b, v_b, Hq)  # -> [seqlen, Hq, D]

        # 分数: [Hq, seqlen]
        scores = torch.einsum("hd,thd->ht", q_b, k_b).to(torch.float32)
        scores = scores * softmax_scale
        # decode 单步通常无需额外掩码（只有过去，没有未来）
        attn = F.softmax(scores, dim=-1).to(dtype)      # [Hq, seqlen]
        out_b = torch.einsum("ht,thd->hd", attn, v_b)   # [Hq, D]
        outs.append(out_b)

    return torch.stack(outs, dim=0)  # [B, Hq, D]


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q: (B, num_heads * head_dim)
        k/v: (B, num_kv_heads * head_dim)
        """

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        # prefill or decode
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = pytorch_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = pytorch_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)

        return o
