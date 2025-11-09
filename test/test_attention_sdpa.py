import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.attention_sdpa import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)


def _reference_varlen(q, k, v, cu_q, cu_k, scale, causal):
    outputs = []
    num_heads = q.size(1)
    num_kv_heads = k.size(1)
    batch = cu_q.numel() - 1
    for i in range(batch):
        qs, qe = int(cu_q[i].item()), int(cu_q[i + 1].item())
        ks, ke = int(cu_k[i].item()), int(cu_k[i + 1].item())
        if qe == qs:
            continue
        q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)
        k_i = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_i = v[ks:ke].transpose(0, 1).unsqueeze(0)
        if num_heads != num_kv_heads:
            repeat = num_heads // num_kv_heads
            k_i = k_i.repeat_interleave(repeat, dim=1)
            v_i = v_i.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(
            q_i, k_i, v_i, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else q.new_zeros((0, num_heads, q.size(-1)))


def _reference_kvcache(q, k_cache, v_cache, scale, causal):
    batch, _, num_heads, _ = q.shape
    num_kv_heads = k_cache.size(2)
    outputs = []
    for i in range(batch):
        q_i = q[i : i + 1].transpose(1, 2)
        k_i = k_cache[i : i + 1].transpose(1, 2)
        v_i = v_cache[i : i + 1].transpose(1, 2)
        if num_heads != num_kv_heads:
            repeat = num_heads // num_kv_heads
            k_i = k_i.repeat_interleave(repeat, dim=1)
            v_i = v_i.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(
            q_i, k_i, v_i, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale
        )
        outputs.append(out.transpose(1, 2))
    return torch.cat(outputs, dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sdpa_kernel requires CUDA")
def test_flash_attn_varlen_matches_reference():
    device = torch.device("cuda")
    num_heads = 2
    num_kv_heads = 1
    head_dim = 3
    seq_lens = [2, 1]
    total_tokens = sum(seq_lens)

    q = (
        torch.arange(total_tokens * num_heads * head_dim, device=device, dtype=torch.float32)
        .view(total_tokens, num_heads, head_dim)
    )
    k = (
        torch.arange(total_tokens * num_kv_heads * head_dim, device=device, dtype=torch.float32)
        .view(total_tokens, num_kv_heads, head_dim)
    )
    v = (
        torch.arange(total_tokens * num_kv_heads * head_dim, device=device, dtype=torch.float32)
        .add(1.0)
        .view(total_tokens, num_kv_heads, head_dim)
    )

    cu_q = torch.tensor([0, seq_lens[0], total_tokens], dtype=torch.int32, device=device)
    cu_k = cu_q.clone()

    output = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=max(seq_lens),
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(seq_lens),
        cu_seqlens_k=cu_k,
        softmax_scale=1.0,
        causal=True,
        block_table=None,
    )

    reference = _reference_varlen(q, k, v, cu_q, cu_k, scale=1.0, causal=True)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sdpa_kernel requires CUDA")
def test_flash_attn_with_kvcache_matches_reference():
    device = torch.device("cuda")
    num_heads = 2
    num_kv_heads = 1
    head_dim = 2
    batch_size = 2
    cache_len = 3

    q = (
        torch.arange(batch_size * num_heads * head_dim, device=device, dtype=torch.float32)
        .view(batch_size, 1, num_heads, head_dim)
    )
    k_cache = (
        torch.arange(batch_size * cache_len * num_kv_heads * head_dim, device=device, dtype=torch.float32)
        .view(batch_size, cache_len, num_kv_heads, head_dim)
    )
    v_cache = (
        torch.arange(batch_size * cache_len * num_kv_heads * head_dim, device=device, dtype=torch.float32)
        .add(2.0)
        .view(batch_size, cache_len, num_kv_heads, head_dim)
    )

    cache_seqlens = torch.tensor([cache_len, cache_len], dtype=torch.int32, device=device)

    output = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=None,
        softmax_scale=1.0,
        causal=True,
    )

    reference = _reference_kvcache(q, k_cache, v_cache, scale=1.0, causal=True)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
