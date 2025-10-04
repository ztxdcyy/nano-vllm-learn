# NanoVLLM 共享内存通信机制分析

## 概述

在NanoVLLM的多GPU分布式推理中，使用共享内存（Shared Memory, SHM）实现进程间通信。这种设计实现了主从架构，其中Rank 0作为控制器，其他Rank作为工作节点。

## 通信架构

### 1. 主从架构设计

- **Rank 0**：控制器进程，负责创建共享内存和发送指令
- **其他Rank**：工作进程，进入监听循环等待指令

### 2. 初始化流程

```python
# Rank 0 进程
self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
dist.barrier()  # 等待其他进程

# 其他Rank进程  
dist.barrier()  # 等待Rank 0创建完成
self.shm = SharedMemory(name="nanovllm")
self.loop()     # 进入监听循环
```

## SHM数据结构

共享内存被设计为一个简单的二进制缓冲区，结构如下：

```
[4字节长度信息] + [序列化数据]
┌─────────────┬─────────────────────────────┐
│  长度(n)    │     序列化数据(pickle)      │
└─────────────┴─────────────────────────────┘
```

### 数据格式说明

- **前4字节**：存储序列化数据的长度（小端序）
- **后续字节**：存储通过pickle序列化的函数调用信息

## 通信协议实现

### 1. 写入过程（Rank 0）

在 `write_shm` 方法中：

```python
def write_shm(self, method_name, *args):
    assert self.world_size > 1 and not self.rank
    data = pickle.dumps([method_name, *args])  # 序列化函数名和参数
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")  # 写入长度信息
    self.shm.buf[4:n+4] = data                   # 写入序列化数据
    for event in self.event:
        event.set()  # 通知其他进程数据已就绪
```

### 2. 读取过程（其他Rank）

在 `read_shm` 方法中：

```python
def read_shm(self):
    assert self.world_size > 1 and self.rank
    self.event.wait()  # 等待Rank 0的通知
    n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取数据长度
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化
    self.event.clear()  # 清除事件，准备接收下一指令
    return method_name, args
```

### 3. 函数调用抽象

在 `call` 方法中实现函数调用的抽象：

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)  # Rank 0写入指令
    method = getattr(self, method_name, None)
    return method(*args)  # 所有Rank执行对应方法
```

## 实际使用示例

### 调用 `run` 方法

当调用 `self.call("run", seqs, True)` 时：

**写入端（Rank 0）**：
- 序列化：`["run", seqs, True]` → 二进制数据
- 写入SHM：长度信息 + 序列化数据

**读取端（其他Rank）**：
- 读取SHM：`method_name = "run", args = [seqs, True]`
- 执行：`self.run(seqs, True)`

## 同步机制

### 1. Barrier同步

- **目的**：确保所有进程在关键操作前达到同步点
- **Rank 0**：创建共享内存后等待其他进程
- **其他Rank**：连接共享内存前确保Rank 0已完成创建

### 2. Event事件通知

- **写入后**：Rank 0通过 `event.set()` 通知数据就绪
- **读取前**：其他Rank通过 `event.wait()` 等待数据
- **读取后**：通过 `event.clear()` 重置事件状态

## 总结

NanoVLLM的SHM通信机制通过简单的二进制协议和同步机制，实现了多GPU进程间的函数调用抽象。这种设计让分布式推理看起来像单进程调用一样简单，同时保持了良好的性能和可靠性。