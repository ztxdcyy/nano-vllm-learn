import argparse
import gc
import os
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def _is_oom(err: Exception) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def _run_single(llm: LLM, input_len: int, num_seqs: int, output_len: int) -> float:
    """Run one generate pass and return elapsed milliseconds."""
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_seqs)
    ]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000


def bench_backend(
    path: str,
    backend: str,
    input_lens: list[int],
    batch_sizes: list[int],
    max_model_len: int,
    enforce_eager: bool,
) -> dict[int, dict[int, float | None]]:
    seed(0)
    results: dict[int, dict[int, float | None]] = {}
    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=max_model_len, attn_backend=backend)
    try:
        # warmup
        # prompt_len=1会触发block_manager要求last_block.hash != -1 的 assert error
        llm.generate(["warmup"], SamplingParams(max_tokens=1))
        for bs in batch_sizes:
            for input_len in input_lens:
                output_len = input_len  # 需求：输出长度与输入相同
                try:
                    ms = _run_single(llm, input_len, bs, output_len)
                    throughput = bs * output_len / (ms / 1000)  # tok/s，仅按输出token计算
                    results.setdefault(bs, {})[input_len] = throughput
                except Exception as err:  # noqa: BLE001
                    if _is_oom(err):
                        results.setdefault(bs, {})[input_len] = None
                        _cleanup()
                    else:
                        raise
    finally:
        # 确保销毁进程组，避免下一个 backend 报 “init_process_group twice”
        try:
            llm.exit()
        except Exception:
            pass
        del llm
        _cleanup()
    return results


def render_table(
    results: dict[str, dict[int, dict[int, float | None]]],
    input_lens: list[int],
    batch_sizes: list[int],
    backends: list[str],
) -> None:
    print("\n" + "=" * 80)
    print("CROSSOVER ANALYSIS (tok/s, output_len=input_len)")
    print("=" * 80)
    for bs in batch_sizes:
        header = f"Batch Size {bs}".ljust(14)
        header += " | " + " | ".join(f"{b:>12}" for b in backends)
        print(header)
        print("-" * len(header))
        for input_len in input_lens:
            row = f"{input_len:>10}"
            for b in backends:
                tp = results.get(b, {}).get(bs, {}).get(input_len)
                row += " | " + (f"{tp:.0f}".rjust(12) if tp is not None else "OOM".rjust(12))
            print(row)
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.expanduser("/root/autodl-tmp/models/Qwen3-0.6B"))
    parser.add_argument(
        "--attn-backend",
        nargs="+",
        default=["flash", "triton", "sdpa.math"],
        choices=["flash", "sdpa", "sdpa.math", "triton"],
    )
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true", help="Force eager mode for debugging.")
    return parser.parse_args()


def main():
    args = parse_args()
    effective_max_len = max(args.input_lens) * 2  # 输入+输出（输出固定等于输入）
    if effective_max_len > args.max_model_len:
        raise ValueError(
            f"max(input_lens)+output_len={effective_max_len} exceeds max_model_len={args.max_model_len}. "
            "请调低 input-lens/output-len 或显式提高 max-model-len（确认模型支持）后再跑 bench。"
        )
    results: dict[str, dict[int, float | None]] = {}
    for backend in args.attn_backend:
        results[backend] = bench_backend(
            args.model_path,
            backend,
            args.input_lens,
            args.batch_sizes,
            args.max_model_len,
            args.enforce_eager,
        )
    render_table(results, args.input_lens, args.batch_sizes, args.attn_backend)


if __name__ == "__main__":
    main()
