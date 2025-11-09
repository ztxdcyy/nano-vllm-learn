from __future__ import annotations

import torch


def online_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax using the online recurrence described in the referenced figure.

    This implements the relations

        m_i = max(m_{i-1}, x_i)
        d'_i = d'_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)

    and finally

        softmax(x)_i = exp(x_i - m_N) / d'_N

    where N is the length of the target dimension.

    Parameters
    ----------
    logits:
        Tensor of arbitrary shape containing the unnormalised scores.
    dim:
        Dimension along which softmax is computed.

    Returns
    -------
    Tensor with the same shape as ``logits`` containing the stable softmax values.
    """
    if logits.numel() == 0:
        return logits.clone()

    # 把需要归一化的dim转置到最后一维
    scores_last_dim = logits.transpose(dim, -1).contiguous()
    prefix_shape = scores_last_dim.shape[:-1]
    target_dim_size = scores_last_dim.shape[-1]

    rows = scores_last_dim.reshape(-1, target_dim_size)
    softmax_rows = torch.empty_like(rows)

    for idx in range(rows.size(0)):
        vec = rows[idx]
        running_max = torch.full((), float("-inf"), dtype=vec.dtype, device=vec.device)
        running_sum = torch.zeros((), dtype=vec.dtype, device=vec.device)
        for value in vec:
            new_max = torch.maximum(running_max, value)
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(value - new_max)
            running_max = new_max
        softmax_rows[idx] = torch.exp(vec - running_max) / running_sum

    return softmax_rows.reshape(*prefix_shape, target_dim_size).transpose(-1, dim)


__all__ = ["online_softmax"]


if __name__ == "__main__":
    # 示例输入: [batch, length, dimension] = [2, 3, 4]
    b, l, d = 2, 3, 4
    torch.manual_seed(42)
    logits = torch.randn(b, l, d)
    
    print("输入张量形状:", logits.shape)
    print("输入张量:\n", logits)
    
    # 在最后一个维度(dimension)上应用online_softmax
    result_dim2 = online_softmax(logits, dim=-1)
    print(f"\n在维度2(dimension)上的softmax结果形状: {result_dim2.shape}")
    print("结果:\n", result_dim2)
    
    # 验证: 每个维度的和应该接近1
    print(f"\n维度2上每行的和: {result_dim2.sum(dim=-1)}")
    
    # 在length维度上应用online_softmax
    result_dim1 = online_softmax(logits, dim=1)
    print(f"\n在维度1(length)上的softmax结果形状: {result_dim1.shape}")
    print(f"维度1上每列的和: {result_dim1.sum(dim=1)}")
    
    # 与PyTorch内置softmax对比
    torch_result = torch.softmax(logits, dim=-1)
    print(f"\n与PyTorch内置softmax的差值: {torch.abs(result_dim2 - torch_result).max().item()}")