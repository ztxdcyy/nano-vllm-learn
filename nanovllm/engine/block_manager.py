from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        # 清空 引用计数和 block 里的内容
        assert block.ref_count == 0
        block.reset()
        # 从 free 中移除，加入 used
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        # 该函数为一条序列 seq 构建它的“逻辑块 → 物理块”的映射表（block_table）。
        # 前提：这条序列之前没有分配过任何物理块。
        assert not seq.block_table

        h = -1                 # parent hash（链式哈希的初值；-1 表示“无/不缓存”）
        cache_miss = False     # 标记：一旦某个块未命中，后续块全部 miss（同一次 prefill 内）

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)    # 得到第 i 个逻辑块里的 token 序列（长度一般是 block_size，最后一块可能不足）

            # 对“满块”计算链式哈希（把前缀的哈希值也编码进去）；“非满块”（最后一块不足 block_size）不参与哈希与缓存
            # 这样：只有内容完整且稳定的块才进入全局缓存，避免部分块导致的重复命中/无效共享
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # 在“全局哈希 → 物理块 id”字典里查找是否已有同样 hash 的 block（prefix caching 的核心）
            # 注意这里包含了前缀 hash，所以只要这个 block能命中，就代表前面的都命中了 
            # 命中就得到可共享的物理块 id（不同会话/请求之间可共享）；否则为 -1
            block_id = self.hash_to_block_id.get(h, -1)

            # 双重确认：即使哈希命中，也再比对一遍 block 内的token 序列，规避哈希碰撞风险
            # （vLLM 也强调哈希仅作键，必要时要防碰撞校验） 
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                # 缓存未命中（或已经进入未命中分支）：从“空闲块池”里拿一个物理块
                # 这里拿队头：free_block_ids[0]；_allocate_block 会把该块从空闲队列搬到“在用集合”，并返回块对象
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：本序列可以直接“映射指向”这个已有物理块，无需重算 K/V
                # 命中一个“满块”意味着这整块 token 的 KV 都可直接复用，等价于 seq 在这块上“已计算”
                seq.num_cached_tokens += self.block_size

                # 如果这个物理块当前已经被别人持有（在 used_block_ids 里），那就是多路共享 → 引用计数 +1
                # （PagedAttention 的“共享页 + 引用计数 + COW”思想）
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 否则说明该物理块目前不在用（只是“被缓存记录”），
                    # 现在把它正式从空闲池“激活”为在用（分配到本序列）——不会复制数据，仍指向同一物理页
                    block = self._allocate_block(block_id)

            # 若本块是“满块”，则更新它的元数据：内容哈希 + token_ids，
            # 并把“哈希 → 物理块 id”的映射写回全局缓存字典，方便后续请求/序列复用
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 无论命中与否，这条序列的逻辑第 i 块，都要在它自己的 block_table 里记录
            # “该逻辑块对应哪个物理块 id”（PagedAttention 的“页表/块表”）
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # len(seq) = 255 256 257
        # 255%256 = 255 !=1 free_block_ids>=0 return True
        # 256%256=0!=1 free_block_ids>=0 return True
        # 257%256=1 判断 free_block_ids 是否大于等于1，也就是是否有新的block供申请？
        # 因为一个token只是占一个slot，而在block-kvcache中，只有block占满了，才会考虑申请新的block
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
