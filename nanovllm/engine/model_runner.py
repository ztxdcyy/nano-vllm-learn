import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

# 初始化流程中先通过 warmup_model 完成 “全局初始化”，
# 再通过 capture_cudagraph 为不同 batch size 预存计算图，
# 最终实现 “无论实际推理用什么 batch size，都能快速复用预存的图”。
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)          # 使用 nccl 建立分布式通信 group
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        # init modelrunner 的时候就会执行 capture_cudagraph，捕捉若干 bs 的计算图，加速推理
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多卡通过 shm+pickle 执行通信
        if self.world_size > 1:
            # rank0 负责创建 shm
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()      # 大哥：到饭店，开好包间以后，等待其余进程
            else:
                dist.barrier()      # 小弟：到饭店了，等大哥开包间吃饭
                # 其余 rank 读取 shm，开始 loop 循环
                self.shm = SharedMemory(name="nanovllm")
                self.loop()     # 无限循环，等待 rank0 指令（从 shm 读指令和参数，直到 exit）

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # 非 rank0 的其他进程，都要跑这个 loop
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            # 就算是 exit，也会先执行 exit 函数，再 break loop
            if method_name == "exit":
                break
    
    # 从 shm 中读取要执行的函数和传递的参数（通过 pickle传递）
    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # 前 4byte 指定了指令+参数的长度n，4后面的根据 n 读取并加载指令和参数
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # rank0也是要干活的！！只是 rank0 额外做一次写 shm 操作
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # 常规 warmup，只要 init model runner，都要走一遍 warmup_model
    def warmup_model(self):
        # 清空 pytorch allocator 占用但是未被使用的 GPU 内存碎片
        torch.cuda.empty_cache()
        # 重新统计下 GPU 最大使用峰值 mem
        torch.cuda.reset_peak_memory_stats()

        # 确保生成的模拟序列既不超过总 token 限制，也不超过最大序列数限制。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)

        # 创建 dummySeqs，里面全是0
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 跑一下
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # 申请一块连续的大 kvcache，然后根据 layerid 绑定到 对应layer的attention 模块的属性中
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config

        # https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # 上一步更新过的
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 计算 kvcache 需要的字节数：2*num_layers*num_kv_heads*head_dim*block_size*torch.dtype
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        # 计算需要的 blocks数量，总共*利用率-所有mem中已用的-peak（allocator历史mem峰值）+ 当前allocator已分配的
        # 主要防止 gpu allocator 可能还需要再用一些内存，所以为了保险，还把这部分也减去了
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 向 GPU 申请一块大 kvcache
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
                                     self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        # kvcache 申请的时候是一个连续的多维的张量，现在需要绑定给指定 layer
        for module in self.model.modules():
            # 假如 module 有 kvcache 属性，则是一个 attn module
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 我们将上面分配的连续的大缓存 self.kv_cache 的指定 layer 切片绑定成该 attn module 类的属性 ———— k_cache & v_cache 
                # 这样 attn module 就可以直接用 kvcache 了，不用来大的再一个个找了。
                # 具体看：nanovllm/layers/attention.py Attention.fwd 
                # k_cache, v_cache = self.k_cache, self.v_cache
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # prepare 主要是创建 tensor， padding，并且传输到 gpu 上
    def prepare_block_tables(self, seqs: list[Sequence]):
        # Python 语法糖：列表推导式
        max_len = max(len(seq.block_table) for seq in seqs)
        # 使用 -1 做 padding
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 创建张量，使用pin memory，可以使用 DMA engine 优化CPU→GPU传输
        # Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices 
        # bandwidth between host memory and device memory is higher if host memory is allocated as page-locked and even higher if in addition it is allocated as write-combining as described in Write-Combining Memory. ———— cuda 编程手册
        # img/pined mem from cuda.png
        # .cuda(non_blocking=True) - 异步传输到GPU，不阻塞CPU
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        # 根据实际在跑的 seqs，初始化一些变量
        for seq in seqs:
            seqlen = len(seq)
            # 收集 seq 中没有被 cached 的 token，注意这里是 seq.num_cached_tokens: 切片
            input_ids.extend(seq[seq.num_cached_tokens:])
            # 生成对应位置的🆔
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # 新的，要计算的q，经典八股之为什么没有 q cache？
            seqlen_q = seqlen - seq.num_cached_tokens
            # 包含 kvcache
            seqlen_k = seqlen
            """
            假设有2个序列：
            序列1：总长度100，已缓存80，新计算20
            序列2：总长度150，已缓存100，新计算50

            # q的累积长度（新token）
            cu_seqlens_q = [0, 20, 70]  # 0→20→70
            max_seqlen_q = 50

            # k的累积长度（所有token）  
            cu_seqlens_k = [0, 100, 250]  # 0→100→250
            max_seqlen_k = 150
            """
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue        # 表示这条序列还没有被分配过 kv block，跳过后续的 slot_mapping 计算
            # 只处理新的token（没有被计算过 kvcache 的），计算 token 对应的物理地址 slot_mapping
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                # 第 i 个 block 在 全局 kvcache 中的起始位置
                start = seq.block_table[i] * self.block_size
                # 假如不是最后一个block，也就是block一定是完整的。在 vLLM 中，只有完整的 block 才可以被 reuse
                if i != seq.num_blocks - 1:
                    end = start + self.block_size           
                # 最后一个block很有可能是不完整的，我们通过last_block_num_tokens计算 end 位置
                else:
                    end = start + seq.last_block_num_tokens         # 在 nanovllm/engine/sequence.py 中定义了last_block_num_tokens属性，最后一个 block 中的 token 个数
                slot_mapping.extend(list(range(start, end)))        # slot_mapping 告诉attention kernel每个token应该存储在KV Cache的哪个位置，对应 prefill 是一连串 token 要写入，所以是 range

        # prefix cache
        # k 的累计前缀和的最后一个元素大于q，也就意味着该 seq 存在完整可复用 kvcache block
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    
            # 检验：假如进入这个分支，assert error 并终止，可以看到在 example（较短 prompt，不存在 cached block） 中是不会进入这个分支的 
            assert False, (
                f"Unexpected state: cu_seqlens_k[-1] ({cu_seqlens_k[-1]}) > "
                f"cu_seqlens_q[-1] ({cu_seqlens_q[-1]}). Check input sequences."
            )
            # 创建 tensor， padding，并且传输到 gpu 上
            block_tables = self.prepare_block_tables(seqs)
        
        # 创建，pin，传输
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # attn kernel 通过 get context 得到上下文，prefill 阶段主要是  flashattention kernel 要用到的一些参数
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)            # 只取 last token
            positions.append(len(seq))
            context_lens.append(len(seq))

            start = seq.block_table[-1] * self.block_size
            offset = seq.last_block_num_tokens
            slot_loc = start + offset - 1
            # print("In Decode phrase, start and offset: ", start, offset)
            slot_mapping.append(slot_loc)

        # 创建 pin 传输
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # attn kernel 通过 get context 得到上下文，decode 阶段主要是 kvcache 和 block table
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 获取正在跑的真实 seq batchsize
            bs = input_ids.size(0)
            context = get_context()
            # 找到第一个 >= 真实bs 的预捕获 bs 对应的 graph（已经捕捉到的）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 刷新、清空、reset
            for k, v in graph_vars.items():
                if k != "outputs":  # outputs 不用管，反正都会覆盖掉的
                    v.zero_()       # 清空但不重新分配
            # 只在前 bs 行写入真实数据
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # run by replay
            graph.replay()
            # 取字典 key="outputs" 对应的值，取前 bs 切片，返回
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # prepare for model_run：prefill or decode
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None        # prepare_sample：return temperatures
        # if decode, use cudagraph
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # 为一组 bs 捕获计算图，buffer 使用最大 bs，并复用 graph mem pool
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # config.max_num_seqs = 512
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # cudagraph 图捕获期间使用的内存必须在图的生命周期内保持有效。这意味着：
        # 图捕获时使用的张量内存地址不能改变，如果使用临时张量，图重放时可能会访问无效内存
        # 必须预先分配足够大的内存来容纳最大可能的batch size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 假设 max_bs = 512, 则 graph_bs = [1,2,4,8,16,32,48, ..., 512]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # 存放所有捕捉的计算图
        self.graphs = {}
        self.graph_pool = None

        # 跑 bs = [512,..., 32,16,8,4,2,1]  为若干常用的 bs 各捕一张图；运行时按大于当前 bs 的最近档位选择一张 graph 来重放。
        # 先跑大的获取最大需要的graph pool，再捕较小的图时使用同一个内存池（with ... self.graph_pool），不再额外申请，从而减少向driver多次申请内存带来的时间开销。
        # https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning?utm_source=chatgpt.com#llm-inference--cuda-graphs
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # nanovllm/utils/context.py 设置这些 global context 变量
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # warmup for cudagraph capture
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # capture cudagraph，捕捉但不执行。
            # with是python中的上下文管理器用法，在这个例子中，启动 cudagraph 捕捉模式，根据传参设定 pool
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 处理 for 循环第一个遍历的时候，还没创建graph pool，根据最大 bs 创建一个graph pool
            # 之后 for 循环遍历的bs会复用该pool
            # https://zhuanlan.zhihu.com/p/700224642
            # https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        
        # 创建一个字典，定义cudagraph需要用到的一系列变量，定义成 ModelRunner的成员变量，这里都是最大 bs
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        # # # 简单打印graph_vars的键值对和检查是否全0
        # # print("=== graph_vars字典内容 ===")
        # # for key, value in self.graph_vars.items():
        # #     print(f"{key}: {value.shape} {value.dtype}")
        # #     print(f"是否全0: {torch.all(value == 0)}")
        # #     print("---")
        # # print("=== 打印结束 ===")
        # 在初始化捕捉图的时候，只有 output 不是全 0