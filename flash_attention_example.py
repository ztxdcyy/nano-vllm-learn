from __future__ import annotations

import math
import torch

def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    block_size: int = 2,
    scale: float | None = None,
) -> torch.Tensor:
    """
    FlashAttention V2 实现 - 外循环遍历Q块，内循环遍历K/V块
    
    q, k, v: [B, L, D] 形状的张量
    返回: [B, Lq, Dv] 形状的张量
    """
    B, Lq, Dq = q.shape
    Bk, Lk, Dk = k.shape
    Bv, Lv, Dv = v.shape
    
    assert B == Bk == Bv, "Batch维度必须一致"
    assert Dq == Dk, "Q/K特征维度必须一致"
    assert Lk == Lv, "K/V序列长度必须一致"
    
    out = torch.zeros((B, Lq, Dv), dtype=q.dtype, device=q.device)
    
    for q_start in range(0, Lq, block_size):
        q_end = min(q_start + block_size, Lq)
        q_blk = q[:, q_start:q_end, :]  # [B, Qs, Dq]
        qs = q_blk.shape[1]
        
        # 累积变量
        running_max = torch.full((B, qs), float("-inf"), dtype=q.dtype, device=q.device)
        d_prev = torch.zeros((B, qs), dtype=q.dtype, device=q.device)
        o_norm = torch.zeros((B, qs, Dv), dtype=q.dtype, device=q.device)
        
        q_idx = torch.arange(q_start, q_end, device=q.device)
        
        for k_start in range(0, Lk, block_size):
            k_end = min(k_start + block_size, Lk)
            k_blk = k[:, k_start:k_end, :]  # [B, Ks, Dq]
            v_blk = v[:, k_start:k_end, :]  # [B, Ks, Dv]
            ks = k_blk.shape[1]
            
            # 计算注意力分数，加上sqrt(d_k)归一化
            scores = torch.matmul(q_blk, k_blk.transpose(-1, -2))  # [B, Qs, Ks]
            dk = q_blk.shape[-1]  # 获取特征维度
            scores = scores / math.sqrt(dk)  # 默认加上sqrt(d_k)归一化
            if scale is not None:
                scores = scores * scale
            
            # 因果掩码
            if causal:
                k_idx = torch.arange(k_start, k_end, device=q.device)
                mask = q_idx.unsqueeze(-1) < k_idx  # [Qs, Ks]
                scores = scores.masked_fill(mask.view(1, qs, ks), float("-inf"))
            
            # 数值稳定的softmax计算
            local_max = scores.max(dim=-1).values  # [B, Qs]
            new_max = torch.maximum(running_max, local_max)
            
            exp_prev = torch.exp(running_max - new_max)
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            
            # 更新累积变量（直接递推已归一化输出）
            d_new = exp_prev * d_prev + exp_scores.sum(dim=-1)
            d_new_safe = d_new.clamp_min(1e-9)
            o_norm = (
                ((d_prev * exp_prev) / d_new_safe).unsqueeze(-1) * o_norm
                + (torch.matmul(exp_scores, v_blk) / d_new_safe.unsqueeze(-1))
            )
            d_prev = d_new
            
            running_max = new_max
        
        # 计算最终输出
        out[:, q_start:q_end, :] = o_norm
    
    return out

def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    block_size: int = 2,
    scale: float | None = None,
) -> torch.Tensor:
    """
    FlashAttention V1 实现 - 外循环遍历K/V块，内循环遍历Q块
    
    q, k, v: [B, L, D] 形状的张量
    返回: [B, Lq, Dv] 形状的张量
    """
    B, Lq, Dq = q.shape
    Bk, Lk, Dk = k.shape
    Bv, Lv, Dv = v.shape
    
    assert B == Bk == Bv, "Batch维度必须一致"
    assert Dq == Dk, "Q/K特征维度必须一致"
    assert Lk == Lv, "K/V序列长度必须一致"
    
    out = torch.zeros((B, Lq, Dv), dtype=q.dtype, device=q.device)
    
    for q_start in range(0, Lq, block_size):
        q_end = min(q_start + block_size, Lq)
        q_blk = q[:, q_start:q_end, :]  # [B, Qs, Dq]
        qs = q_blk.shape[1]
        
        # 累积变量
        running_max = torch.full((B, qs), float("-inf"), dtype=q.dtype, device=q.device)
        denom = torch.zeros((B, qs), dtype=q.dtype, device=q.device)
        numer = torch.zeros((B, qs, Dv), dtype=q.dtype, device=q.device)
        
        q_idx = torch.arange(q_start, q_end, device=q.device)
        
        for k_start in range(0, Lk, block_size):
            k_end = min(k_start + block_size, Lk)
            k_blk = k[:, k_start:k_end, :]  # [B, Ks, Dq]
            v_blk = v[:, k_start:k_end, :]  # [B, Ks, Dv]
            ks = k_blk.shape[1]
            
            # 计算注意力分数，加上sqrt(d_k)归一化
            scores = torch.matmul(q_blk, k_blk.transpose(-1, -2))  # [B, Qs, Ks]
            dk = q_blk.shape[-1]  # 获取特征维度
            scores = scores / math.sqrt(dk)  # 默认加上sqrt(d_k)归一化
            if scale is not None:
                scores = scores * scale
            
            # 因果掩码
            if causal:
                k_idx = torch.arange(k_start, k_end, device=q.device)
                mask = q_idx.unsqueeze(-1) < k_idx  # [Qs, Ks]
                scores = scores.masked_fill(mask.view(1, qs, ks), float("-inf"))
            
            # 数值稳定的softmax计算
            local_max = scores.max(dim=-1).values  # [B, Qs]
            new_max = torch.maximum(running_max, local_max)
            
            exp_prev = torch.exp(running_max - new_max)
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            
            # 更新累积变量（分子/分母分别累积）
            denom = exp_prev * denom + exp_scores.sum(dim=-1)
            numer = exp_prev.unsqueeze(-1) * numer + torch.matmul(exp_scores, v_blk)
            
            running_max = new_max
        
        # 计算最终输出
        out[:, q_start:q_end, :] = numer / denom.clamp_min(1e-9).unsqueeze(-1)
    
    return out


def main() -> None:
    """使用统一的输入格式测试v1和v2实现"""
    # 使用range(24)生成输入数据，view为(2,3,4)形状
    q = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    k = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    v = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    
    print("输入形状:")
    print(f"q: {q.shape}")
    print(f"k: {k.shape}")
    print(f"v: {v.shape}")
    print()
    
    # 测试v1实现
    print("测试FlashAttention V1:")
    out_v1 = flash_attention_v1(q, k, v, causal=True, block_size=2)
    print(f"V1输出形状: {out_v1.shape}")
    print("V1输出:")
    print(out_v1)
    print()
    
    # 测试v2实现
    print("测试FlashAttention V2:")
    out_v2 = flash_attention_v2(q, k, v, causal=True, block_size=2)
    print(f"V2输出形状: {out_v2.shape}")
    print("V2输出:")
    print(out_v2)
    print()
    
    # 对比两个版本的差异
    diff = (out_v1 - out_v2).abs().max().item()
    print(f"V1与V2的最大绝对误差: {diff:.6f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()