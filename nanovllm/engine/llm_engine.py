import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 获取启动配置参数
        config = Config(model, **config_kwargs)
        self.ps = []        # 存储TP子进程列表
        self.events = []       # 进程间同步事件列表 
        # print(self.ps, self.events)

        # 通过torch.multiprocessing spawn形式，启动单机TP
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # 每个子进程运行ModelRunner实例
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 主进程实例化 ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 将tokenizer的eos_token_id同步到config.eos中
        config.eos = self.tokenizer.eos_token_id
        # 主进程实例化调度器
        self.scheduler = Scheduler(config)
        # 清理操作，通过钩子实现，暂时不看
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 初始化Seq，传入prompt和sampling_params
        seq = Sequence(prompt, sampling_params)
        # 在scheduler的waiting队列加入当前seq
        self.scheduler.add(seq)

    def step(self):
        # 1️⃣ 调度：由 scheduler 决定本轮要跑哪些序列、处于哪个阶段（prefill / decode）
        seqs, is_prefill = self.scheduler.schedule()
        # 2️⃣ 推理：把调度结果交给 model_runner 进行一次前向，返回新生成的 token_ids
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 3️⃣ 后处理：token_ids这里确保了只会吐出一个token。我们将新的token加到相应的seq里并且释放遇到EoS的seq
        self.scheduler.postprocess(seqs, token_ids)
        # 4️⃣ 收集已完成序列：只把is_finished的序列输出给上层
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 5️⃣ 统计 token 数：prefill 阶段为正（用于吞吐计算），decode 阶段为负（便于区分）
        # 用正负号区分阶段，方便外层统计吞吐
        if is_prefill:
            # prefill 阶段：返回所有序列 token 总数（正数）
            num_tokens = sum(len(seq) for seq in seqs)
            print("num_tokens: ", num_tokens)
        else:
            # decode 阶段：返回序列条数的相反数（负数），后面通过判断正负就可以知道是pd哪个阶段了，一种优化写法
            num_tokens = -len(seqs)
        # num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        # 返回：已完成序列列表 以及 本轮处理的 token 数（供外层进度条和吞吐计算使用）
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[str]:
        # tqdm：python进度条库，用于显示进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 创造和prompts等长的sampling_params列表，用于存储每个prompt的采样参数
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 将prompt和对应的sp打包成seq，添加到调度器中
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}

        prefill_throughput = decode_throughput = 0.
        # 核心循环，不断调用step，直到所有seq都完成
        while not self.is_finished():
            t = perf_counter()

            # ❗❗❗ 执行step，返回output(seq_id, 当前已生成的完整 token_ids)和num_tokens
            output, num_tokens = self.step()

            # 可视化进度条，显示当前进度和当前的prefill和decode吞吐量。这里从代码推断的，没有实际跑，等我改掉attn！！！
            # 这里会根据num_tokens的正负号来区分是prefill还是decode阶段，从而计算吞吐量。减少了一个bool位的占用
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 根据 seq_id 汇总output，得到outputs
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        # 调用 transformers.tokenizer.decode 根据 token_ids 解码得到真正的 token 继续收集到 outputs 里面
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
