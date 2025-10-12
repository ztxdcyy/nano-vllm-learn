from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs        
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # waiting和running都为空时，调度器判断请求完成
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 最开始都加入waiting等待调度
        self.waiting.append(seq)

    # seqs, is_prefill = self.scheduler.schedule()
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []         # step内成功调度的序列
        num_seqs = 0                # 已调度序列计数器
        num_batched_tokens = 0      # 当前batch中（所有running-seq）所有序列需要计算的token总数。“购物车的实时重量显示器”，告诉调度器"还能装多少"
        # waiting队列不为空，且seq数量没有超限，就往running队列里塞seq，直到塞爆：max_num_tokens&&max_num_seqs&&max_model_len;
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
        # 假如有调度成功的队列，立即进入return prefill，后面的decode就不会管了
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            # 从running队列头部弹出一个seq
            seq = self.running.popleft()
            # can_append稍微有点啰嗦，主要是判断是不是return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
            # 当不能append的时候，假如有running，则驱逐一个running
            while not self.block_manager.can_append(seq):
                if self.running:
                    # running弹出头部后，还有running，则弹出running的尾巴的seq，且用preempt实现状态转换+内存释放
                    # 这里体现了FCFS，即优先保证running的第一个seq被服务。当取出的队首seq无法被服务时，本次while循环将抢占running最后一个seq的内存。
                    self.preempt(self.running.pop())
                else:
                    # running弹出头部后，没有running了（也就是弹出的seq 是最后一个running seq）
                    # 此时意味着服务不了，当前seq也只能释放掉
                    self.preempt(seq)
                    break
            else:
                # 内存足以append的时候，执行may_append
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        # 确保至少有一个成功调度的seq，假如没有，这里可能会asserterror。
        # 我觉得有可能的，假设没有任何成功调度的seq。
        # 大while中你取了running的头部seq
        # 当无法can_append时，while小循环内running为空进入else分支，驱逐当前seq并且break。
        assert scheduled_seqs
        # 按照原顺序，将调度成功的序列放回running头部
        # extendleft：将list中的元素逐个加到running的left
        # 比如 scheduled_seqs = [seq0, seq1, seq2]
        # 假如不做reserved，则running将变成[seq2, seq1, seq0,old_running]，调度的顺序就反了
        # 所以要做reserved，running = [seq0, seq1, seq2, old_running]
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 驱逐，输入一个seq，更新状态为waiting
        seq.status = SequenceStatus.WAITING
        # 清空内存
        self.block_manager.deallocate(seq)
        # 重新回到waiting队首
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # 上一步modelrunner确保了只吐出一个token（要是投机采样的话这里还得改?）
            seq.append_token(token_id)
            # 检查该token_id，看看是不是结束了，需要释放资源
            # case1:不忽略 eos 且 tokenid 是 eos
            # case2:计算完的token数量达到最大token
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 更新状态：Finished；释放循环内seqs的内存，在running中移除该seq。
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
