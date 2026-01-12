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
    num_seqs: int,
    output_len: int,
    max_model_len: int,
    enforce_eager: bool,
) -> dict[int, float | None]:
    seed(0)
    results: dict[int, float | None] = {}
    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=max_model_len, attn_backend=backend)
    try:
        # warmup
        llm.generate(["warmup"], SamplingParams(max_tokens=1))
        for input_len in input_lens:
            try:
                results[input_len] = _run_single(llm, input_len, num_seqs, output_len)
            except Exception as err:  # noqa: BLE001
                if _is_oom(err):
                    results[input_len] = None
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
    results: dict[str, dict[int, float | None]],
    input_lens: list[int],
    num_seqs: int,
    output_len: int,
) -> None:
    print("\n" + "=" * 80)
    print("CROSSOVER ANALYSIS")
    print("=" * 80)
    print(
        f"{'Input Len':>10} | {'Flash (ms)':>11} | {'Flash tp':>11} | "
        f"{'SDPA (ms)':>10} | {'SDPA tp':>10} | {'Winner':>8} | {'Speedup':>8}"
    )
    print("-" * 96)
    for input_len in input_lens:
        flash_t = results.get("flash", {}).get(input_len)
        sdpa_t = results.get("sdpa", {}).get(input_len)
        decode_tokens = num_seqs * output_len

        if flash_t is None and sdpa_t is None:
            winner, speedup = "N/A", "N/A"
            flash_tp = sdpa_tp = "N/A"
        elif flash_t is None:
            winner, speedup = "SDPA", "N/A"
            flash_tp = "OOM"
            sdpa_tp = f"{decode_tokens / (sdpa_t / 1000):.0f}" if sdpa_t else "OOM"
        elif sdpa_t is None:
            winner, speedup = "Flash", "N/A"
            sdpa_tp = "OOM"
            flash_tp = f"{decode_tokens / (flash_t / 1000):.0f}" if flash_t else "OOM"
        else:
            winner = "Flash" if flash_t <= sdpa_t else "SDPA"
            speedup = max(flash_t, sdpa_t) / min(flash_t, sdpa_t)
            speedup = f"{speedup:.2f}x"
            flash_tp = f"{decode_tokens / (flash_t / 1000):.0f}"
            sdpa_tp = f"{decode_tokens / (sdpa_t / 1000):.0f}"

        def fmt(v: float | None) -> str:
            return "OOM" if v is None else f"{v:.3f}"

        print(
            f"{input_len:>10} | {fmt(flash_t):>11} | {flash_tp:>11} | "
            f"{fmt(sdpa_t):>10} | {sdpa_tp:>10} | {winner:>8} | {speedup:>8}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.expanduser("/root/autodl-tmp/models/Qwen3-0.6B"))
    parser.add_argument("--attn-backend", nargs="+", default=["flash", "sdpa"], choices=["flash", "sdpa"])
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 1536, 2048, 2560, 3072, 3584],
    )
    parser.add_argument("--num-seqs", type=int, default=32, help="Batch size for each input_len test.")
    parser.add_argument("--output-len", type=int, default=512, help="Max tokens to sample for each prompt.")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true", help="Force eager mode for debugging.")
    return parser.parse_args()


def main():
    args = parse_args()
    effective_max_len = max(args.input_lens) + args.output_len
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
            args.num_seqs,
            args.output_len,
            args.max_model_len,
            args.enforce_eager,
        )
    # Only render table when both backends are present
    if {"flash", "sdpa"}.issuperset(set(results.keys())):
        render_table(results, args.input_lens, args.num_seqs, args.output_len)
    else:
        # Fallback: print the single backend numbers
        backend = next(iter(results))
        print(f"\n[{backend}] results (ms):")
        for input_len in args.input_lens:
            val = results[backend].get(input_len)
            if val is None:
                line = "OOM"
            else:
                decode_tokens = args.num_seqs * args.output_len
                tp = decode_tokens / (val / 1000)
                line = f"{val:.3f} ms, {tp:.0f} tok/s"
            print(f"  input_len={input_len}: {line}")


if __name__ == "__main__":
    main()
