import argparse
import time

import torch

from moe_linear_fusion import fused_moe, baseline_moe


@torch.inference_mode()
def run_once(fn, hidden_states, w1, w2, topk_weights, topk_ids, topk):
    return fn(hidden_states, w1, w2, topk_weights, topk_ids, topk)


def benchmark(fn, hidden_states, w1, w2, topk_weights, topk_ids, topk, warmup, iters, sync_fn):
    # warmup
    for _ in range(warmup):
        run_once(fn, hidden_states, w1, w2, topk_weights, topk_ids, topk)
        sync_fn()

    start = time.perf_counter()
    for _ in range(iters):
        run_once(fn, hidden_states, w1, w2, topk_weights, topk_ids, topk)
        sync_fn()
    end = time.perf_counter()
    return end - start


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused_moe vs baseline_moe.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="float16", help="float16 or float32")
    parser.add_argument("--num_tokens", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--up_dim", type=int, default=4096)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    M = args.num_tokens
    K = args.hidden_dim
    up_dim = args.up_dim
    E = args.num_experts
    topk = args.topk

    hidden_states = torch.randn(M, K, device=device, dtype=dtype)
    # w1: [E, up_dim * 2, K], w2: [E, K, up_dim] (down projection)
    w1 = torch.randn(E, up_dim * 2, K, device=device, dtype=dtype) * 0.02
    w2 = torch.randn(E, K, up_dim, device=device, dtype=dtype) * 0.02

    # random topk_ids and weights
    topk_ids = torch.randint(0, E, (M, topk), device=device)
    topk_weights = torch.softmax(torch.randn(M, topk, device=device, dtype=dtype), dim=-1)

    # sync helper
    sync_fn = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    # correctness check (single run)
    out_fused = run_once(fused_moe, hidden_states, w1, w2, topk_weights, topk_ids, topk)
    out_base = run_once(baseline_moe, hidden_states, w1, w2, topk_weights, topk_ids, topk)
    max_diff = (out_fused - out_base).abs().max().item()

    fused_time = benchmark(fused_moe, hidden_states, w1, w2, topk_weights, topk_ids, topk, args.warmup, args.iters, sync_fn)
    base_time = benchmark(baseline_moe, hidden_states, w1, w2, topk_weights, topk_ids, topk, args.warmup, args.iters, sync_fn)

    fused_tput = (M * args.iters) / fused_time
    base_tput = (M * args.iters) / base_time

    print(f"device={device}, dtype={dtype}, M={M}, K={K}, up_dim={up_dim}, E={E}, topk={topk}")
    print(f"max diff (fused vs baseline): {max_diff:.6f}")
    print(f"fused_moe:    {fused_time:.4f}s total, {fused_tput:.2f} tokens/s")
    print(f"baseline_moe: {base_time:.4f}s total, {base_tput:.2f} tokens/s")


if __name__ == "__main__":
    main()
