# nano-vllm-learn
# 特点
1. 详细的注释，方便学习
2. without-flash-attn的实现：我觉得flash-attn在最开始安装环境时候时候需要编译比较花时间，就想做一个wo-flash-attn的版本。main分支已经合并了完整实现，并且改了bench，下面说；wo_flash_attn分支记录了开发过程。


# 采用SDPA代替flash-attn：
* 主要工作量是如何从block-tables里取到对应的kvcache。
* 主要踩坑的地方是和cudagraph的兼容问题：可以看这个[commit](https://github.com/ztxdcyy/nano-vllm-learn/commit/2f1a0ae2df9f7729494c5c70caf010dd786d2b5e)
1. 在捕捉cudagraph的时候禁止 host 侧的操作，具体可以看kaichao这篇文章，所以原来的tolist、item这些都不能用
2. 还是捕捉cudagraph的时候（ModelRunner.capture_cudagraph），dummyinput在构造的时候，context_lens=torch.zeros(...)。和我原来的一个assert冲突了`assert isinstance(max_seq_len, int) and max_seq_len > 0, "max_seq_len 必须是正整数"`，添加了更鲁棒的判断来兼容capture cudagraph跑dummyinput的场景。
* 踩坑的地方+1：传递backend的时候一直没传hf_config，导致一直跑的都是flash_attn也就是attn_sdpa.py完全没被用上！所以基本上测出来吞吐没变！！！（🥲尴尬……）

在这套代码里，模型构造用的是读的是 Qwen3Config 实例（dataclass：config.hf_config），Qwen3DecoderLayer/Qwen3Attention 只看它：attn_backend=getattr(config, "attn_backend", "flash")。LLMEngine.Config 是运行时包装，模型只会拿到hf_config（Qwen3Config）不会把 大的运行时Config 传进去。所以不把 attn_backend 写回 hf_config，模型侧永远拿不到你传的后端，默认为 flash。传递过去只是为了让 HF config 携带这个自定义字段，使模型能读到。


# bench

```
(nano_venv) root@autodl-container-b95c4d8452-4b3d06c8:~/workspace/nano-vllm-learn# python bench_my.py --attn-backend flash sdpa
`torch_dtype` is deprecated! Use `dtype` instead!
^[[A^[[B
================================================================================
CROSSOVER ANALYSIS
================================================================================
 Input Len |  Flash (ms) |    Flash tp |  SDPA (ms) |    SDPA tp |   Winner |  Speedup
------------------------------------------------------------------------------------------------
       512 |    3075.008 |        5328 |  42203.928 |        388 |    Flash |   13.72x
      1024 |    4267.898 |        3839 |  64877.147 |        253 |    Flash |   15.20x
      1536 |    5468.262 |        2996 |  88115.733 |        186 |    Flash |   16.11x
      2048 |    6651.592 |        2463 | 111999.628 |        146 |    Flash |   16.84x
      2560 |    7869.468 |        2082 | 136555.871 |        120 |    Flash |   17.35x
      3072 |    9091.268 |        1802 | 161280.169 |        102 |    Flash |   17.74x
      3584 |   10328.921 |        1586 | 186514.650 |         88 |    Flash |   18.06x
```

居然没有OOM？不会吧。我们在bench.py里为什么OOM了？

## 删除official bench的随机性，添加命令行参数`--attn-backend`
我修改了[official bench代码](bench.py)的随机性，原来他的代码是定下一个max-input-len和max-output-len，然后`randint(100, max)`随机取输入输出长度，我觉得还是定下来我比较安心，就把randint删掉了。

同时新增了`attn-backend`的命令行参数，用来指定使用sdpa还是flash-attn，默认是flash。

跑下来结果：
```
(nano_venv) root@autodl-container-b95c4d8452-4b3d06c8:~/workspace/nano-vllm-learn# python bench.py --attn-backend flash
`torch_dtype` is deprecated! Use `dtype` instead!
Total: 262144tok, Time: 64.33s, Throughput: 4075.22tok/s
(nano_venv) root@autodl-container-b95c4d8452-4b3d06c8:~/workspace/nano-vllm-learn# python bench.py --attn-backend sdpa
`torch_dtype` is deprecated! Use `dtype` instead!
Total: 262144tok, Time: 64.34s, Throughput: 4074.61tok/s
```

## 新增了表格对比
为了对比我的sdpa backend和flash-attn backend性能差了多少我新增了一个[`bench_my.py`](bench_my.py)，它能生成一个表格，对比latency和throughput以及加速倍数，参考的是[here](https://github.com/Wenyueh/MinivLLM/blob/main/benchmark_decoding.py)

```
(nano_venv) root@autodl-container-b95c4d8452-4b3d06c8:~/workspace/nano-vllm-learn# python bench.py --model-path /root/autodl-tmp/models/Qwen3-0.6B   --attn-backend flash sdpa  --num-seqs 256 --input-lens 1024   --output-len 1024
`torch_dtype` is deprecated! Use `dtype` instead!

================================================================================
CROSSOVER ANALYSIS
================================================================================
 Input Len |  Flash (ms) |    Flash tp |  SDPA (ms) |    SDPA tp |   Winner |  Speedup
------------------------------------------------------------------------------------------------
      1024 |   64340.523 |        4074 |  64022.840 |       4095 |     SDPA |    1.00x
```
当然也可以传入一个list，会生成真正的表格【有个小限制，input+output不要超过max-model-len，否则模型都是胡言乱语】：
```
(nano_venv) root@autodl-container-b95c4d8452-4b3d06c8:~/workspace/nano-vllm-learn# python bench_my.py 
`torch_dtype` is deprecated! Use `dtype` instead!

================================================================================
CROSSOVER ANALYSIS
================================================================================
 Input Len |  Flash (ms) |    Flash tp |  SDPA (ms) |    SDPA tp |   Winner |  Speedup
------------------------------------------------------------------------------------------------
       512 |    3093.975 |        5295 |   3085.169 |       5311 |     SDPA |    1.00x
      1024 |    4248.941 |        3856 |   4244.249 |       3860 |     SDPA |    1.00x
      1536 |    5438.311 |        3013 |   5442.521 |       3010 |    Flash |    1.00x
      2048 |    6606.076 |        2480 |   6604.461 |       2481 |     SDPA |    1.00x
      2560 |    7773.946 |        2108 |   7800.910 |       2100 |    Flash |    1.00x
      3072 |    8971.982 |        1826 |   9001.928 |       1820 |    Flash |    1.00x
      3584 |   10226.610 |        1602 |  10228.934 |       1602 |    Flash |    1.00x
```

## 分析
可以看到SDPA和flash-attn在端到端的情况下基本没有差距，sdpa本质也是采用flash-attn思想优化的kernel。
Attention 机制哪家强？SDPA、FlashAttention、xFormers、手动实现全面对比 - 一条放浪不羁的爬虫的文章 - 知乎
https://zhuanlan.zhihu.com/p/1898470649938293363


我还需要压测一下，看看SDPA什么时候会OOM。我跑了[大佬的代码](https://github.com/Wenyueh/MinivLLM/blob/main/benchmark_prefilling.py)发现对比triton实现，fa还是在显存上有显著优势，在长序列场景下，只有flash依旧活着！！！

```
================================================================================
CROSSOVER ANALYSIS
================================================================================
   Seq Len |   Naive (ms) |   Flash (ms) |     Winner |    Speedup
--------------------------------------------------------------------------------
        16 |        0.029 |        0.053 |      Naive |      1.80x
        32 |        0.028 |        0.056 |      Naive |      1.97x
        48 |        0.028 |        0.098 |      Naive |      3.44x
        64 |        0.030 |        0.059 |      Naive |      1.97x
        80 |          OOM |        0.055 |      Flash |        N/A
        96 |          OOM |        0.053 |      Flash |        N/A
       112 |          OOM |        0.053 |      Flash |        N/A
       128 |          OOM |        0.054 |      Flash |        N/A
       192 |          OOM |        0.062 |      Flash |        N/A
       256 |          OOM |        0.070 |      Flash |        N/A
       512 |          OOM |        0.134 |      Flash |        N/A
      1024 |          OOM |        0.343 |      Flash |        N/A

================================================================================
KERNEL LAUNCH ANALYSIS
================================================================================

For 2 sequences × 60 tokens:
  Naive Triton grid:    (2, 32)
  Naive total kernels:  64

  Flash Attention grid: (2, 32, 2)
  Flash total kernels:  128

  Ratio: Flash launches 2.0x more kernels

  Each kernel launch has ~5-20μs overhead
  Extra overhead: ~640μs = 0.64ms
```

# Future Plan(nano-moe coming soon)

因为一直在研究moe推理优化，所以想在nanovllm上实现下面这几个特性，把这个仓库慢慢转变成`nano-moe`哈哈哈😄：

-[ ] 支持dpsk-moe

  - [ ] Nano vllm triton mla

  - [ ] Nano vllm triton moe kernel fusion  https://zhuanlan.zhihu.com/p/21251657579

- [x] Nano vllm triton paged-attn

- [ ] Nano vllm eplb

- [ ] Nano vllm shared-expert-overlap





