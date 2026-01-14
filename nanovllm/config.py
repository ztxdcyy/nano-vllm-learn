import os
from dataclasses import dataclass
from transformers import AutoConfig


VALID_ATTN_BACKENDS = ("flash", "sdpa", "sdpa.math", "triton")


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    attn_backend: str = "flash"

    def __post_init__(self):
        assert self.attn_backend in VALID_ATTN_BACKENDS
        # Keep backward compatibility with the old "sdpa" flag.
        if self.attn_backend == "sdpa":
            self.attn_backend = "sdpa.math"
        # 让 HF config 携带自定义字段（后端、块大小）供模型读取
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 将注意力后端写入 HF config，供模型构造时读取
        setattr(self.hf_config, "attn_backend", self.attn_backend)
        setattr(self.hf_config, "kvcache_block_size", self.kvcache_block_size)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
