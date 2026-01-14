import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.benchmark as benchmark

# 检查 CUDA 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Warning: CUDA is not available. Benchmarking on CPU will not show FlashAttention advantages.")

def bench_sdpa(batch_size, num_heads, seq_len, head_dim, backend, is_causal=True):
    # 构造输入数据 (B, H, L, D)
    # 使用 float16 因为 FlashAttention 主要针对半精度优化
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    desc = f"SDPA {backend}"
    sub_label = f"L={seq_len}, H={num_heads}"

    # 使用 sdpa_kernel 上下文管理器强制指定后端
    with sdpa_kernel(backend):
        t = benchmark.Timer(
            stmt="F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)",
            setup="import torch.nn.functional as F",
            globals={'q': q, 'k': k, 'v': v, 'is_causal': is_causal},
            label="SDPA Speed Comparison",
            sub_label=sub_label,
            description=desc
        )
        return t.blocked_autorange()

# 测试参数配置
configs = [
    (16, 12, 1024, 64), # 标准场景
    (8, 12, 4096, 64),  # 长序列场景，此时 Math 后端显存压力剧增
]

results = []
for b, h, l, d in configs:
    # 测试 Math 后端
    results.append(bench_sdpa(b, h, l, d, SDPBackend.MATH))
    # 测试 FlashAttention 后端 (如果硬件支持)
    try:
        results.append(bench_sdpa(b, h, l, d, SDPBackend.FLASH_ATTENTION))
    except RuntimeError:
        print(f"FlashAttention not supported on this device for L={l}")

# 打印对比表格
compare = benchmark.Compare(results)
compare.print()