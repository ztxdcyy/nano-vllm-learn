ModelRunner.__init__ 会设置 NCCL 进程组、切换默认 dtype/device、构造 Qwen3ForCausalLM 并用 load_model 装载权重，然后实例化 Sampler。接着依次执行 warmup_model→allocate_kv_cache→（若允许）capture_cudagraph，最后在多卡场景下为 rank0/rank>0 建立共享内存通信并启动 rank>0 的事件循环。

warmup_model 用最大模型长度和总 token 限制生成一批 dummy Sequence，跑一次 run(..., is_prefill=True) 以编译 kernel、拉起 cudnn/cublas 缓存，再清理残余显存。

allocate_kv_cache 根据 GPU 利用率目标估算能分配的 KV block 数，申请一块 [2, n_layer, n_block, block_size, n_kv_head, head_dim] 的连续显存，并把每一层的切片绑定到注意力模块的 k_cache/v_cache 属性。

capture_cudagraph 仅为解码阶段准备：用一组持久化张量和多组 batch size（[1,2,4,8,16,...]）捕获前向计算图，并复用同一个 graph_pool 减少显存碎片；上下文信息通过 set_context 写入全局 Context。

推理时调用 run(seqs, is_prefill)：prefill 阶段使用 prepare_prefill，解码阶段使用 prepare_decode。两种模式都把需要的张量固定到 pinned memory 后再异步拷入 GPU，run_model 决定走 eager 还是复用捕获图，rank0 再用 Sampler 抽样新 token 并返回。