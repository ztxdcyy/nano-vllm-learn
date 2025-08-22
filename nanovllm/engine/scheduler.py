from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs         # 默认512？并发能打这么大？一个小demo也这么能打？
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # waiting和running都为空时，调度器完成
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 最开始都加入waiting等待调度
        self.waiting.append(seq)

    # seqs, is_prefill = self.scheduler.schedule()
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []         # step内成功调度的序列
        num_seqs = 0                # 已调度序列计数器
        num_batched_tokens = 0      # 当前batch中（所有running-seq）所有序列的token总数。“购物车的实时重量显示器”，告诉调度器"还能装多少"
        # waiting队列不为空，且seq数量没有超限，就往running队列里塞seq，直到塞爆：max_num_tokens;max_num_seqs;max_model_len;
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 超出上限，跳出prefill循环
            # 1. 超出max_num_batched_tokens
            # 2. 内存不足，block_manager分配失败
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 调用blockmanager做内存分配
            self.block_manager.allocate(seq)
            # 计算num_batched_tokens：序列长度-已缓存token数 也就是本轮实际需要计算的tokne数量
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 更新seq状态为RUNNING，当前seq来自于waiting[0]
            seq.status = SequenceStatus.RUNNING
            # seq从waiting⬅️弹出
            self.waiting.popleft()
            # seq进入running尾巴
            self.running.append(seq)
            # 又新增一员已调度序列啦～
            scheduled_seqs.append(seq)
        # 假如有调度成功的队列，立即进入prefill
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
