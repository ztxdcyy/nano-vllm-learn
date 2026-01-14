import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def moe_linear_kernel(
    a_ptr,  # [M, K]
    w_ptr,  # [E, N, K]
    c_ptr,  # [M_total, N]
    offsets_ptr,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_cm,
    stride_cn,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    batch_start = tl.load(offsets_ptr + expert_id)
    batch_end = tl.load(offsets_ptr + expert_id + 1)
    M_len = batch_end - batch_start

    num_pid_m = tl.cdiv(M_len, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if (pid_m >= num_pid_m) or (pid_n >= num_pid_n):
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M_len
    token_idx = batch_start + offs_m

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_SIZE_K):
        offs_k = k0 + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + token_idx[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = (
            w_ptr
            + expert_id * stride_we
            + offs_n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk
        )
        b = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b)

    acc = acc.to(tl.float32)
    c_ptrs = c_ptr + (batch_start + offs_m)[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def moe_linear_launcher(hidden_states, expert_weights, expert_offsets, *, block_m=32, block_n=128, block_k=32):
    """
    Triton kernel launcher: hidden_states [M, K], expert_weights [E, N, K],
    sorted_ids [M*topk], expert_offsets [E+1] -> output [M*topk, N].
    """
    M, K = hidden_states.shape
    E, N, K_w = expert_weights.shape
    assert K == K_w, "K dimension mismatch"

    out = torch.empty((hidden_states.size(0), N), device=hidden_states.device, dtype=hidden_states.dtype)

    grid = (
        E,
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    moe_linear_kernel[grid](
        hidden_states,
        expert_weights,
        out,
        expert_offsets,
        hidden_states.stride(0),
        hidden_states.stride(1),
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        out.stride(0),
        out.stride(1),
        K=K,
        N=N,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )
    return out


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [E, up_dim * 2, K]
    w2: torch.Tensor,  # [E, hidden_dim, up_dim]
    topk_weights: torch.Tensor,  # [M, topk]
    topk_ids: torch.Tensor,  # [M, topk]
    topk: int,
) -> torch.Tensor:
    device = hidden_states.device
    dtype = hidden_states.dtype

    M = hidden_states.size(0)  # token 数
    E, up2, K = w1.shape
    up_dim = up2 // 2  # gate + up

    # 按专家分组的下标和 offset
    flat_ids = topk_ids.flatten()
    sorted_ids = flat_ids.argsort()
    expert_counts = torch.bincount(flat_ids, minlength=E)
    expert_offsets = torch.cat([torch.tensor([0], device=device), expert_counts.cumsum(0)])

    # 将 hidden_states 扩展成 [M*topk, K]，再按 sorted_ids 重排
    expanded_hidden = hidden_states.repeat_interleave(topk, dim=0)
    permuted_hidden = expanded_hidden[sorted_ids]

    # 上行：用 Triton kernel 做 batched matmul（对应 w1）
    up_buffer = moe_linear_launcher(
        permuted_hidden,
        w1,
        expert_offsets=expert_offsets,
    )

    # 激活：拆成 gate/up，做 SiLU，再与 up 相乘
    gate_state, up_state = up_buffer.chunk(2, dim=-1)
    activated = F.silu(gate_state, inplace=True) * up_state

    # 下行：再次按专家分块做 matmul（对应 w2）
    down_dim = K
    down_buffer = moe_linear_launcher(
        activated,
        w2,
        expert_offsets=expert_offsets,
    )

    # 把排序后的结果按 token 聚合回去，并乘以 gate 的 topk 权重
    token_ids = torch.arange(M * topk, device=device) // topk
    perm_token_ids = token_ids[sorted_ids]
    perm_weights = topk_weights.flatten()[sorted_ids].unsqueeze(1)
    weighted = down_buffer * perm_weights

    output = torch.zeros((M, down_dim), device=device, dtype=dtype)
    output.scatter_add_(0, perm_token_ids.unsqueeze(1).expand_as(weighted), weighted)
    return output


def baseline_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [E, up_dim * 2, K]
    w2: torch.Tensor,  # [E, hidden_dim, up_dim]
    topk_weights: torch.Tensor,  # [M, topk]
    topk_ids: torch.Tensor,  # [M, topk]
    topk: int,
) -> torch.Tensor:
    """最朴素的 for-loop baseline：按专家分片，逐个专家做两层线性并加权回写。"""
    device = hidden_states.device
    dtype = hidden_states.dtype

    M = hidden_states.size(0)
    E, up2, K = w1.shape
    up_dim = up2 // 2

    flat_ids = topk_ids.flatten()
    flat_weights = topk_weights.flatten()
    sample_indices = torch.arange(M, device=device)[:, None].expand(-1, topk).flatten()

    outputs = torch.zeros((M, K), device=device, dtype=dtype)

    for e in range(E):
        mask = flat_ids == e
        if not mask.any():
            continue

        tokens = sample_indices[mask]
        weights = flat_weights[mask]

        # 专家前向：上行 -> SiLU * up -> 下行
        up = hidden_states[tokens] @ w1[e].T
        gate_state, up_state = up.chunk(2, dim=-1)
        activated = F.silu(gate_state, inplace=False) * up_state
        down = activated @ w2[e].T  # [num_tokens_for_expert, K]

        weighted = down * weights.unsqueeze(1)
        outputs.index_add_(0, tokens, weighted)

    return outputs


if __name__ == "__main__":
    # 一个可运行的示例，隐藏维度 8，up_dim 16，3 个专家，topk=2
    torch.manual_seed(0)
    hidden_dim = 8
    up_dim = 16
    num_tokens = 4
    topk = 2
    num_experts = 3

    hidden_states = torch.arange(1, num_tokens + 1, dtype=torch.float32).unsqueeze(1).repeat(1, hidden_dim)
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]])
    topk_weights = torch.rand(num_tokens, topk)

    # w1: [E, up_dim*2, hidden_dim], w2: [E, hidden_dim, up_dim]
    w1 = torch.randn(num_experts, up_dim * 2, hidden_dim) * 0.1
    w2 = torch.randn(num_experts, hidden_dim, up_dim) * 0.1

    out_fused = fused_moe(hidden_states, w1, w2, topk_weights, topk_ids, topk)
    out_baseline = baseline_moe(hidden_states, w1, w2, topk_weights, topk_ids, topk)

    print("fused shape:", out_fused.shape, "baseline shape:", out_baseline.shape)
    print("max diff:", (out_fused - out_baseline).abs().max().item())
    print("baseline sample (token 0):", out_baseline[0])
