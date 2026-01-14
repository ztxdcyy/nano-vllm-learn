import os
import time
from random import randint, seed
import argparse
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn-backend",
        type=str,
        default="flash",
        choices=["flash", "sdpa", "sdpa.math", "triton"],
        help="Attention backend to use (flash, sdpa.math, or triton)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed(0)
    num_seqs = 64
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("/root/autodl-tmp/models/Qwen3-0.6B/")
    # 新增了attn-backend，在这里传递进config就可以调用不同的attn实现了
    llm = LLM(path, enforce_eager=False, max_model_len=4096, attn_backend=args.attn_backend)

    # 修改掉随机性，生成的都是定长
    prompt_token_ids = [[randint(0, 10000) for _ in range(max_input_len)] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_ouput_len) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
