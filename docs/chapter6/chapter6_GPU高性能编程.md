# 第 6 章 GPU 高性能编程

## 本章学习目标

读完本章应能做到三件事：

- 用 CUDA event 写一个可复用的 benchmark，并用 `torch.profiler` 定位到具体 CUDA kernel 与时间分布。
- 写出覆盖 elementwise / reduction / row-overflow / matmul tiling 四类模式的 Triton kernel，并解释为什么每种模式对应不同的 `program_id` / `tl.arange` / mask 策略。
- 从 PTX 文本中读出 `.target`、`%ctaid.x`、`%tid.x`、`ld.global.v4.b32` 等信号，并把它们对应到上面的 Triton 抽象。

## 本章主线

GPU 高性能编程先从一条可复用的排查链开始：

1. 用 benchmark 得到端到端时间。
2. 用 profiler 找到实际执行的 CUDA kernel 和时间分布。
3. 根据瓶颈判断优化方向：减少 kernel launch、减少 HBM 往返、提高 coalescing、增加 tile 复用，或交给编译器融合。
4. 在 PyTorch builtin、`torch.compile`、Triton、CUDA C++ 和 PTX 之间选择维护成本最低、收益足够明确的工具。

这条排查链的认识起点是 GPU 执行模型。一次 kernel 的 wall-clock $T_{\text{kernel}}$ 由两条物理上限共同决定：

$$T_{\text{kernel}} = \max\!\left( T_{\text{memory}},\; T_{\text{compute}} \right) + T_{\text{launch}} + T_{\text{sync}}$$

其中 $T_{\text{memory}}$ 是 HBM / shared memory / register 三级存储的访存上限，$T_{\text{compute}}$ 是 Tensor Core 与 CUDA core 的算术上限，$T_{\text{launch}}$ 是 host 端 launch 开销（与算子复杂度无关），$T_{\text{sync}}$ 是 block / grid 之间的同步开销。benchmark 固定输入形状与 warm-up 后测端到端时间，profiler 将时间拆回具体 kernel，Triton / PTX 实验再检验线程映射、向量化和内存复用是否改变瓶颈；因此每个优化结论都绑定到测量变量和适用 workload。

> [!IMPORTANT]
> 上述公式是 roofline 的标准结构：kernel 实际性能不会超过 $\max(T_{\text{memory}}, T_{\text{compute}})$ 这条上界，下界由 $T_{\text{launch}} + T_{\text{sync}}$ 给出。判断一个 kernel 是 memory-bound 还是 compute-bound，等价于比较 $T_{\text{memory}}$ 与 $T_{\text{compute}}$ 谁占优——这正是 §6.1 末尾「算术强度」段的物理意义。

读完本章后，应当能把一段高层 PyTorch 代码拆成三层问题：Python 表达式触发了哪些 kernel；这些 kernel 在 GPU 上受哪些硬件约束；需要下探时，Triton program、thread block 和 PTX 信号如何对应到实际执行。

硬件基础参见 [第 5 章 §5.2 GPU 的执行模型 SM（流式多处理器）](../chapter5/chapter5_GPU和GPU相关优化.md) 与 [第 5 章 §5.3 GPU 的内存模型](../chapter5/chapter5_GPU和GPU相关优化.md)；本章侧重编程模型、benchmark / profile 流程和 Triton / PTX 的工程落点。

正文中的代码片段保留核心结构，省略的 import、helper 和检查函数不影响主线。

## 6.1 GPU 编程模型和硬件约束

本节先看两个层面的约束：程序员写下的并行抽象，以及硬件能同时驻留多少工作、从哪一级内存取数。读完应能在 SM / warp / memory hierarchy 的视角下读懂后续 benchmark 和 profiler 的输出。

![图 6.1-1 GPU memory hierarchy](images/6-1-1-gpu-memory-hierarchy.png)

*图 6.1-1 GPU memory hierarchy*

图 6.1-1 左侧是 SM 及其内部 registers、L1/shared memory 与芯片级 L2，右侧是容量更大的 HBM。优化 kernel 时最常见的收益来自把数据从 HBM 读入更近的层级后复用，或减少中间结果写回 HBM 的次数。

硬件速查表把 A100、H100 和 B200 放在一起，用来建立数量级直觉。后面的 trace 和 PTX 观察主要按 B200 / Blackwell 语境理解。

| 指标 | A100 | H100 | B200 |
| --- | ---: | ---: | ---: |
| SM 数 | 108 | 132 | 148 |
| 每 SM register | 256 KB | 256 KB | 256 KB |
| 每 SM L1 + shared memory | 192 KB | 256 KB | 256 KB |
| L2 cache | 40 MB | 50 MB | 单颗 GB200 / B200 GPU（全封装，含 2 个 GB100 die）L2 = 126 MB（[NVIDIA Blackwell tuning guide §1.4.2.2](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)）；折算每 GB100 die ≈ 63 MB |
| HBM 容量与类型 | 80 GB HBM2e | 80 GB HBM3 | 180 GB HBM3e（[NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) 公布的可寻址上限；HGX B200 / GB200 NVL72 公开 datasheet 取同一数值） |
| HBM 带宽量级 | 2 TB/s | 3.35 TB/s | 8 TB/s |

*表 6.1 A100/H100/B200 硬件数量级*

> [!NOTE]
> 表里的“每 SM register 256 KB”指 SM 内部 register file 的物理容量：每个 SM 有 64K 个 32-bit register，即 $65536 \times 4\ \text{B} = 256\ \text{KB}$ 。它和“每个 thread 最多使用 255 个 32-bit register”是两个不同维度的上限——前者决定一个 SM 上能并发驻留多少 thread block，后者决定单个 thread 的寄存器压力。后面算 occupancy 时两个数字都会用到。
>
> “每 SM L1 + shared memory”一行是统一 cache 的总容量。在 B200（compute capability 10.0）上，这 256 KB 中能配置成 shared memory 的部分是 228 KB，单个 thread block 最多申请 227 KB。

带宽账本比容量账本更直接影响 kernel 优化。各层带宽的差距由物理位置和总线宽度决定，不只是容量大小：

- **Register**：位于 SM 内部，与 ALU 同 cycle，无总线仲裁。每周期可同时服务多个 warp 的访问，这是为什么 register bandwidth 达到数百 TB/s。
- **L1 + shared memory**：位于 SM 内部，但需要经过 32 路 bank 仲裁器。每周期每个 bank 只能服务一次访问（不同地址时），仲裁上限决定有效吞吐。
- **L2 cache**：芯片级共享，通过 crossbar interconnect 跨 SM 访问。比 register 慢一个数量级，但容量大一到两个数量级。
- **HBM**：芯片外，通过 1024-bit 物理接口（如 HBM3 / HBM3e）传输，物理距离和通道数决定带宽上限；容量最大，但延迟与带宽都比片上小一个数量级。

Registers 局部性最强，shared memory 适合一个 thread block 内协作，L2 是芯片级缓存，HBM 容量最大但访问代价也最高。

| 带宽层级 | A100 | H100 | B200 |
| --- | ---: | ---: | ---: |
| Register bandwidth\* | ~116 TB/s | ~401 TB/s | ~447 TB/s |
| L1 cache + shared memory bandwidth | ~19 TB/s | ~33 TB/s | ~19 TB/s |
| L2 cache bandwidth | ~5-8 TB/s | ~12 TB/s | ~9 TB/s |
| HBM bandwidth | 2 TB/s | 3.35 TB/s | 8 TB/s |

*表 6.2 A100/H100/B200 内存带宽数量级*

\* Register bandwidth 是按 SM 数、时钟频率和寄存器端口宽度换算出来的量级估计，用于和下面三级带宽做数量级对比。
> [!NOTE]
> Hopper 与 Blackwell 在 memory hierarchy 上各加了一层上表没有列出的资源，对默认编程模型（CUDA C++、Triton、TorchInductor）不可见：
>
> - **Tensor Memory（TMEM，Blackwell 数据中心型号 B200 / GB200）**：在 Tensor Core 旁新增的张量专用内存，位于 register 与 shared memory 之间，`tcgen05.mma` 的累加器可以直接驻留其上。TMEM 256 KB/SM、按 128 lane × 512 column 的 32-bit 单元组织（$128 \times 512 \times 4\ \text{B} = 256\ \text{KB}$）；CUDA C++ 与 Triton 路径下编译器自动管理，写 PTX 时 alloc / dealloc 指令才直接暴露给程序员。
> - **Thread Block Cluster（Hopper 引入，Blackwell 沿用）**：把多个 thread block 编为一个 cluster，cluster 内 block 可跨 SM 直接访问彼此的 distributed shared memory；Hopper 的 portable cluster size 最多 8 个 SM，Blackwell 在显式设置 `cudaFuncAttributeNonPortableClusterSizeAllowed` 后可扩展到 16 个 SM。CUDA C++ 与 Triton 路径下编译器与库自动使用，写 PTX 时 cluster 指令直接暴露给程序员。
>
> 本章后续讨论的算子写法不依赖这两层资源；它们对 Hopper / Blackwell 的最优性能至关重要，但属于 kernel-level 优化的延伸话题，超出本章主线。

CUDA / Triton 的基础并行模型可以写成三层：

```text
grid（一次 kernel launch 的所有工作）
  -> thread block / CTA（一组共享同一片 shared memory 的线程）
    -> thread（在一小块数据上执行指令）
```

`thread block` 在 PTX 和 profiler 语境里也常称为 `CTA`（Cooperative Thread Array）。一个 thread block 会被调度到某个 SM 上执行；block 内线程可以通过该 SM 上的 shared memory、warp shuffle 或编译器生成的 reduction 逻辑共享中间状态。

Elementwise GeLU 这类操作几乎可以按元素独立处理，thread 视角已经够用：每个 thread 读一个元素、做一组算术、写回一个元素，thread 之间没有通信，访存账本就是「输入读 + 输出写」各一次。

Softmax 和 matmul 会立刻需要 block 视角。第一性原理：softmax 的输出 $y_i = \exp(x_i - \max_j x_j) / \sum_j \exp(x_j - \max_j x_j)$ 需要整行最大值与整行分母，这两个 reduction 必须经过 warp / block 内的通信（warp shuffle 或 shared memory），单 thread 视角无法完成；matmul 的输出 $C[m, n]$ 依赖整行 $A[m, :]$ 与整列 $B[:, n]$，如果不做 tile 复用，朴素路径的访存账本是 $O(MNK)$ reads + $MN$ writes，每个 thread block 必须把数据保留在片上才能把访存降到 $O(MK + KN)$。这两类算子的共同点是 block 内的 thread 必须共享状态，因此 thread block / CTA 成为最自然的编程单位。

Triton 的自然思维单位就是”一个 program 负责一个 block 级任务”——把 reduction 通信与 tile 复用直接收进 program 内。

Warp 是硬件执行层最重要的补充抽象。一个 warp 由 32 个 thread 组成，同一个 warp 内的线程以 lockstep 方式执行同一条指令。若同一 warp 内线程走不同分支，硬件通常要顺序执行多条路径，这会造成 control divergence。

Occupancy 有两个常见含义：

- **Warp occupancy**：一个 SM 上同时驻留多少 warp。寄存器、shared memory 和线程数都会限制它。从物理约束推导：每 SM register file 容量 = $64\,\text{K} \times 4\,\text{B} = 256\,\text{KB}$ 是常数；每个 thread 占用的 register 数 × 驻留 thread 数 = register 上限；因此高 register 占用直接降低 warp 数。低 occupancy 不自动代表坏结果：如果 thread coarsening 让每个 thread 做更多有效工作，每个 warp 独立完成的指令更多，较低 occupancy 也可能换来更高吞吐。Occupancy 是 latency hiding 的必要条件，不是充分条件——没有足够多 warp，HBM 等长延迟无法被其他 warp 的算术指令掩盖；反之，warp 多并不保证吞吐高。
- **Block occupancy**：thread block 被分批调度到 SM 上时是否均衡。B200 有 148 个 SM，若只启动 160 个 block，第一波能填满 148 个 SM，第二波只剩 12 个 block，大量 SM 会闲置。物理上，硬件每周期把每个 block 调度到完整的一个 SM 上、block 内 thread 必须共享同一个 SM 的所有资源，因此 block 数超过 SM 数时一定出现尾波。

一个小算例可以把 warp occupancy 的含义固定下来。假设一个 thread block 有 128 个 thread，每个 thread 使用 160 个 registers，那么一个 block 需要 $128 \times 160 = 20480$ 个 registers。若一个 SM 大约有 65536 个 registers 可用（A100 / H100 / B200 公开数值一致），同一个 SM 最多同时驻留 $\lfloor 65536 / 20480 \rfloor = 3$ 个 block。

每个 block 有 $128 / 32 = 4$ 个 warps，所以一共是 12 个 resident warps。若硬件上限是 64 个 warps/SM，这个 kernel 的 warp occupancy 约为 $12 / 64 \approx 18.75\%$。这个数说明寄存器用量已经限制了并发 warp 数；后续优化判断还要结合每个 thread 做了多少有效工作、是否能隐藏 HBM 访问延迟，以及 Tensor Core 是否吃满。

![图 6.1-2 wave quantization](images/6-1-2-wave-quantization.png)

*图 6.1-2 wave quantization*

图 6.1-2 的横轴是矩阵尺寸，两条散点是同一 GEMM 在各尺寸下的实测性能，曲线呈周期性锯齿。取 $256 \times 128$ 的输出 tile：尺寸为 1792 时 tile 数是 $(1792 / 256) \times (1792 / 128) = 7 \times 14 = 98$ ；尺寸增到 1793 后两个方向都要向上取整，tile 数跳到 $8 \times 15 = 120$ 。

A100 有 108 个 SM，120 个 tile 无法在一波内全部执行，尾波只剩 12 个 block，其余 SM 空转，性能曲线上就出现周期性的下跌。

调 block size、grid size 和 thread coarsening 时，需要同时看每个 block 的工作量和总 block 数对 SM 数的整除关系。

Shared memory 和 HBM 还有两类不同的访存坑：

- **Bank conflict** 发生在 shared memory。从硬件结构推导：shared memory 分成 32 个 bank，每个 bank 宽 4 B，每周期每个 bank 只能被一个 thread 访问（不同地址时），即仲裁上限 = $32 \times 4\ \text{B} = 128\ \text{B/cycle}$；同地址可走广播路径，不构成冲突。bank 映射每 128 B 循环一次，连续 float32 元素的 bank 编号是 $\mathrm{bank}(i) = i \bmod 32$。若一个 warp 的多个线程在同一周期访问同一个 bank 的不同地址，这些访问会被串行化，等效吞吐按冲突倍数下降。
- **Memory coalescing** 发生在 global memory / HBM 访问。物理依据：HBM cache line = 128 B = 32 threads × 4 B / thread，warp 内 32 个线程各取 4 B 连续地址时，正好凑成 128 B，硬件可以合并成一条 cache-line transaction；若同一拍访问矩阵不同 row 的同一列，地址跨度通常很大，一次 load 会拆成多条 transaction，cache-line 利用率下降，有效带宽随之降低。

Bank conflict 常出现在 tile 被写入 shared memory 后又按另一种方向读取时。按行连续写入通常很顺，但转置读取或按列读取可能让同一个 warp 的多个 thread 落到同一个 bank。常见缓解方式是 swizzling：保持逻辑坐标不变，只改变 shared memory 中的物理下标，例如把某一维映射成 `row ^ col` 这类按位异或形式。这样相邻 thread 的访问会被打散到更多 bank；读取时使用同一套映射还原逻辑位置。

算术强度（arithmetic intensity）把这些硬件约束压成一个判断：

$$I = \frac{\text{FLOPs}}{\text{bytes}}$$

它是 roofline 模型的核心变量。当 $I > I_{\text{ridge}} = \text{peak FLOP/s} / \text{peak bandwidth}$ 时，kernel 是 compute-bound，$T_{\text{compute}}$ 主导 wall-clock；当 $I < I_{\text{ridge}}$ 时是 memory-bound，$T_{\text{memory}}$ 主导。

按这一判据：逐元素 GeLU、ReLU、加法通常 $I$ 极低（每元素 1 FLOP 左右、4 B 读写），容易被 HBM 往返和 kernel launch 限制；matmul 通过 tile 复用可以把 $I$ 提高到与 tile size 相关——tile 越大，每个 HBM 读入的字节服务更多乘加，$I$ 越高，向 compute-bound 边界靠拢。这是为什么 §6.5 的 matmul tiling 直接决定了算子落到 roofline 的哪个象限。

## 6.2 Benchmark 和 profiler 的工作流

本节走完一次最小排查循环：先写一个可复用的 benchmark 函数测端到端时间，再用 `torch.profiler` 把这段端到端时间拆到具体 CUDA kernel。读完应能解释为什么 benchmark 和 profiler 必须成对使用，以及 `cutlass3x_sm100_simt_sgemm_64x64x16` 这类 kernel 名字里每一段分别对应什么信息。

### 6.2.1 Benchmark 流程

本小节用一个最小 benchmark 函数（CUDA event 计时 + warmup + 多次 trial + `torch.cuda.synchronize`）测端到端时间，并用矩阵乘法扫一组维度观察耗时从「近似常数」过渡到「三次方增长」的拐点。读完后应能自己写一个可复用 benchmark 并解释每一行的作用。

```python
def run_operation2(dim: int, operation: Callable) -> Callable:
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())
    return lambda: operation(x, y)

def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    times: list[float] = []
    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        run()
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    return mean(times)
```

Warmup 处理第一次运行中的编译、缓存和初始化开销。CUDA 同步处理异步提交问题：CPU 侧代码返回不代表 GPU 已完成 kernel。CUDA event 计时能更贴近 GPU 执行窗口，避免把 Python 侧调度时间混进主要数字。

用这个流程测矩阵乘法，可以得到随维度变化的数量级。

```python
results = {}
for dim in [256, 512, 1024, 2048, 4096, 8192]:
    results[dim] = benchmark(
        run_operation2(dim=dim, operation=lambda a, b: a @ b)
    )
```

| 矩阵维度 `dim` | 平均时间（ms） |
| --- | ---: |
| 256 | 0.6149 |
| 512 | 0.5928 |
| 1024 | 0.5909 |
| 2048 | 0.7010 |
| 4096 | 2.5596 |
| 8192 | 17.6036 |

*表 6.3 矩阵乘法 benchmark 结果（单卡 B200）*

小尺寸区间没有立刻呈现理想的三次增长。把端到端时间拆成两部分：

$$T(N) = T_{\text{launch}} + T_{\text{dispatch}} + c \cdot N^3$$

$T_{\text{launch}}$ 是 CPU 端 kernel launch 开销（与算子复杂度无关，约几 μs 量级），$T_{\text{dispatch}}$ 是 cuBLAS heuristic / kernel 选择与第一次调度的固定开销，$c$ 是与算子实现、SM 数、Tensor Core 利用率相关的常数。当 $N^3$ 小时（如 $N = 256$），$T_{\text{launch}} + T_{\text{dispatch}}$ 占主导，$T(N) \approx T_{\text{launch}}$；当 $N^3$ 充分大（如 $N = 8192$），$c \cdot N^3$ 占主导，曲线逼近三次方。

尺寸变大后，$T_{\text{launch}}$ 占比迅速衰减，$T(N)$ 的斜率逐步与 $c \cdot N^3$ 的导数对齐。表 6.3 的 $0.59\text{ms} \to 17.6\text{ms}$ 跨越约 30 倍，与 $N$ 从 $256$ 增到 $8192$（尺寸 ×32）对应的 $(8192/256)^3 = 32^3 = 32768$ 倍理论放大相比，实际只放大约 30 倍——这意味着大尺寸下 cuBLAS 的 kernel 利用率受限于 HBM 带宽与 SM 数，已接近 $T_{\text{memory}}$ 的 roofline 上界（详见 §6.1 末段）。Benchmark 给出真实机器上的拐点，把 $N^3$ 的复杂度公式落到具体硬件常量上。

### 6.2.2 Profiler 看到实际 kernel

Profiler 把高层 PyTorch 调用和实际 CUDA kernel 连接起来。下面的辅助函数先 warmup，再用 PyTorch profiler 采集 CUDA 活动，并按 CUDA 时间排序输出表格。

```python
def profile(run: Callable, num_warmups: int = 1):
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        run()
        torch.cuda.synchronize()

    return prof.key_averages().table(
        sort_by="cuda_time_total",
        max_name_column_width=100,
        row_limit=10,
    )
```

对矩阵加法，Python 里只是 `a + b`，底层会分派到 elementwise CUDA kernel。

| profile 目标 | 主要 CUDA 信号 | Self CUDA time total |
| --- | --- | ---: |
| `add(dim=2048)` | `vectorized_elementwise_kernel...CUDAFunctor_add...` | 4.960 us |

*表 6.4 矩阵加法 profiler 摘要（单卡 B200）*

对矩阵乘法，kernel 名称会暴露更多实现信息。

| profile 目标 | 主要 CUDA kernel 信号 | Self CUDA time total |
| --- | --- | ---: |
| `matmul(dim=2048)` | `cutlass3x_sm100_simt_sgemm...64x64x16...` | 329.345 us |
| `matmul(dim=128)` | `cutlass3x_sm100_simt_sgemm...32x32x16...` | 4.544 us |

*表 6.5 矩阵乘法 profiler 摘要（单卡 B200）*

`cutlass3x` 指 CUTLASS 3.x（NVIDIA 维护的 CUDA 线性代数模板库），`sm100` 表示目标架构是 Blackwell 数据中心 SKU（即 B200 / GB200），`simt` 表示该 kernel 走 SIMT 路径，由普通 CUDA core 执行逐元素乘加，`sgemm` 表示单精度 float32 GEMM，`64x64x16` 或 `32x32x16` 是 tile 形状。相同 Python 表达式在不同形状下可能调用不同 kernel，因此性能分析不能只停在 `a @ b` 这一层。

## 6.3 Kernel fusion：GeLU 作为最小案例

本节用 GeLU 作为最小案例，对比朴素 PyTorch、PyTorch builtin 和 `torch.compile` 三种实现，并解释为什么 fusion 能从「多个 kernel 多次 HBM 往返」压成「一个 kernel 一次往返」。读完应能用 benchmark + profiler 的证据链判断一个算子是否值得 fusion。

```python
def naive_gelu(x: torch.Tensor):
    return 0.5 * x * (
        1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x))
    )

def builtin_gelu(x: torch.Tensor):
    return torch.nn.functional.gelu(x, approximate="tanh")
```

`naive_gelu` 里的乘法、加法、立方和 `tanh` 都是高层 PyTorch op。若每个 op 各自形成 kernel，中间结果会多次读写 HBM。`builtin_gelu` 是 PyTorch 内置融合实现，通常能用更少 kernel 完成同一件事。

第三条路线是把朴素函数交给 `torch.compile`：

```python
compiled_gelu = torch.compile(naive_gelu)
check_equal_1d(naive_gelu, compiled_gelu)
```

三个版本都计算相同函数，但耗时差异很大。下表是 `run_operation1(dim=16384, ...)`（即 16384 × 16384 输入）在单卡 B200 上的实测结果；不同 GPU 与不同输入规模会改变绝对数。

| 实现 | benchmark 平均时间 | profiler 主要信号 | Self CUDA time total |
| --- | ---: | --- | ---: |
| `naive_gelu` | 3.758 ms | 多个逐元素 PyTorch/CUDA kernel | 3.385 ms |
| `builtin_gelu` | 0.667 ms | `GeluCUDAKernelImpl` | 305.409 us |
| `torch.compile(naive_gelu)` | 0.939 ms | `triton_poi_fused_add_mul_tanh_0` | 342.848 us |

*表 6.6 GeLU 三种实现的 benchmark/profiler 对比（`dim=16384`，单卡 B200）*

逐元素链条太碎时，先找 PyTorch builtin；没有合适 builtin 时，试 `torch.compile`。在这个案例里，编译器把朴素计算图收敛成一个 Triton fused kernel，性能接近内置实现，维护成本远低于手写 kernel。内置 GeLU 仍然更快——自动编译提供的是强基线，不保证总能超过专门优化过的库 kernel。

Fusion 的收益来自数据路径的物理约束。从朴素 GeLU 的 op 链出发（`0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x³)))`），可以拆出至少 4 个 HBM-bound PyTorch op：加法（`x + 0.044715 * x³`）、乘法（乘 `0.79788456`）、tanh 近似（`tanh`）、外层加乘。每个 op 在朴素实现中各形成独立 kernel、读一次输入张量、写一次输出张量，因此朴素路径的 HBM 访问次数约 $4 \times (N^2 \text{ reads} + N^2 \text{ writes}) = 8 N^2$ 次；理想 fused 实现把整个公式收进同一个 kernel，每个元素只读一次 $x$、只写一次 $y$，即 $2 N^2$ 次。理论上 fused vs 朴素约 4× 访存上限差距——与 §6.4.2 softmax 的访存账本属于同一类推理。

融合版本把多个逐元素计算留在同一个 kernel 内，理想情况下每个元素只读一次、写一次。表 6.6 里 `builtin_gelu` 与 `compiled_gelu` 相对 `naive_gelu` 实际只快约 5.6× / 4.0×——比理论 4× 略好，因为朴素路径额外承担 kernel launch 与调度开销；理论 4× 给出的是访存账本的极限，实际加速还包含 launch overhead 的消除。

## 6.4 Triton：从 elementwise 到 row-overflow

本节用三个递进的 Triton 例子展示 block 级 program 的写法：GeLU（§6.4.1，一段连续向量一个 program）、softmax（§6.4.2，一行一个 program，reduction 在 program 内完成）、row sum（§6.4.3，一行太长时 program 内循环 tile）。读完后应能用 `tl.program_id` / `tl.arange` / mask 三件套解释不同模式的 program 划分策略；matmul 把这套二维 tile 思想推到矩阵乘，在 §6.5 展开。

> [!TIP]
> **Triton program 与 CUDA 线程的对应**：在 Triton 里写 `tl.program_id(axis=0)` 拿到的就是 CUDA PTX 中的 `%ctaid.x`（block id）；`tl.arange(0, BLOCK)` 是该 program 内部要处理的一段 tile；kernel 编译时 Triton 会自动把这段 tile 拆成 warp × thread 的向量化指令，程序员不必直接管 thread id（对应 `%tid.x`）。这是 Triton 与手写 CUDA 的最大区别——把 block 级逻辑显式写出，把 thread 级调度交给编译器。

### 6.4.1 Triton GeLU：一段连续向量一个 program

本小节看最简单的 Triton 写法：输入是一段连续向量，输出也是一段连续向量，最自然的切分就是「一个 program 负责 BLOCK_SIZE 个连续元素」。读完应能解释 wrapper 中的 grid 形状、`BLOCK_SIZE` 常量和 kernel 中的 mask 分别在做什么。

```python
def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    num_elements = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](
        x, y, num_elements, BLOCK_SIZE=block_size
    )
    return y
```

`[(num_blocks,)]` 是 Triton grid shape，表示启动多少个 program；`BLOCK_SIZE` 是单个 program 一次处理的向量宽度。两者分别回答“开多少个小队”和“每个小队处理多宽”。

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    tl.store(y_ptr + offsets, y, mask=mask)
```

这段 kernel 展示了大多数 Triton 入门代码的骨架：用 `tl.program_id` 找到当前 program，构造一组向量 offset，用 mask 处理末尾越界，加载一段连续数据，在片上完成计算，再写回 HBM。

| trace 信号 | 数值 / 含义 |
| --- | --- |
| `num_elements` | 8192 |
| `BLOCK_SIZE` | 1024 |
| `num_blocks` | 8 |
| PTX 目标 | `.target sm_100a` |

*表 6.7 Triton GeLU PTX 摘要*

### 6.4.2 Triton softmax：一行一个 program

Softmax 比 GeLU 多了 reduction。每个输出元素依赖整行最大值和整行分母，因此自然的切分是让一个 program 负责一行。本小节把 row max、减 max、`exp`、row sum、normalize 这五步收进同一个 program，并对照朴素实现的访存账本说明 fusion 的理论加速上限。

![图 6.4-1 Triton softmax row program](images/6-4-1-triton-softmax.png)

*图 6.4-1 Triton softmax row program*

图 6.4-1 左侧 `pid=1` 选择 row 1，右侧同一个 program 先 `tl.load` 整行，再做 `x - tl.max(x)` 减最大值（数值稳定性）、`tl.exp` 取指数、`tl.sum` 求分母和 normalize，最后 `tl.store` 回输出。row 之间没有共享状态，所以 grid 可以是一维的 `(M,)`。

```python
def triton_softmax(x: torch.Tensor):
    y = torch.empty_like(x)
    M, N = x.shape
    block_size = triton.next_power_of_2(N)
    triton_softmax_kernel[(M,)](
        x_ptr=x,
        y_ptr=y,
        x_row_stride=x.stride(0),
        y_row_stride=y.stride(0),
        num_cols=N,
        BLOCK_SIZE=block_size,
    )
    return y
```

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride,
                          num_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_row = tl.load(
        x_start_ptr + col_offsets,
        mask=col_offsets < num_cols,
        other=float("-inf"),
    )

    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    y_start_ptr = y_ptr + row_idx * y_row_stride
    tl.store(y_start_ptr + col_offsets, y_row, mask=col_offsets < num_cols)
```

`other=float("-inf")` 让越界列在 row max 和 `exp` 中保持安全：masked 位置不会影响最大值，指数后也不会贡献有效概率。这个 mask 处理是 Triton 代码里处理非整齐尺寸的常见模式。

Kernel 内部和朴素 softmax 的数学结构很像，但访存结构完全不同。朴素 PyTorch 会经历 row max、减 max、`exp`、row sum、除法等多个 op，每一步都可能形成独立 kernel 和中间张量。

把每一步的访存账本展开（行数 $M$，列数 $N$）：

| 步骤 | HBM reads | HBM writes |
| --- | ---: | ---: |
| `x_max = x.max(dim=1)` | $MN$ | $M$ |
| `x = x - x_max[:, None]` | $MN + M$ | $MN$ |
| `numerator = torch.exp(x)` | $MN$ | $MN$ |
| `denominator = numerator.sum(dim=1)` | $MN$ | $M$ |
| `y = numerator / denominator[:, None]` | $MN + M$ | $MN$ |
| 合计 | $5MN + 2M$ | $3MN + 2M$ |

合计访存 $5MN + 2M + 3MN + 2M = 8MN + 4M$。理想情况下每个元素只读一次、写一次，只需 $2MN$ 次，两者相差约 4 倍——这就是朴素 softmax 在访存账本上的理论加速上限。

Triton fused softmax 把一行读入后，把 max / sum reduction 和 normalize 都留在同一个 program 内完成，把整条流水线收敛到每行约一次读加一次写。这 4 倍就是它的理论加速上限：实际加速取决于 reduction 是否被 warp shuffle / shared memory 高效实现、是否触发 register spilling，以及最后一次 store 是否真的只有 $MN$ 次写。

### 6.4.3 Triton row sum：行太长时分 tile 累加

一行能放进一个 block 时，softmax 写法很直接。若一行长于 block size，就需要让同一个 program 在行内循环多个 tile。本小节用 row sum 作为最小例子：每个 lane 保持一个 accumulator，遍历 tile 后再用 `tl.sum` 做一次 reduction；这里的 tile 在 program 内部循环，和 grid 中的多个 block 是两层概念。

![图 6.4-2 Triton row sum tiled accumulation](images/6-4-2-triton-row-sum.png)

*图 6.4-2 Triton row sum tiled accumulation*

图 6.4-2 中前两轮 tile 更新每个位置的 `acc`，最后一轮用 mask 跳过越界列，再通过 `tl.sum(acc)` 得到一个标量输出。这里的 tile 属于同一个 program 内部循环，和 grid 中的多个 block 是两层概念。

```python
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    result = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, result)
```

GeLU 的 block 之间完全独立；softmax 的一行必须在一个 program 内形成共同统计量；行太长时，program 内还要显式循环 tile。这个模式会在 matmul 中扩展成二维 tile 复用。

## 6.5 Matmul tiling、PTX 和工具选择

本节先把前面的 block / tile 思想推到矩阵乘：用一个 program 负责 $C$ 的一个 $(\mathrm{BLOCK\_M}, \mathrm{BLOCK\_N})$ 输出 tile，沿 $K$ 维循环加载 $A$ 和 $B$ 的 tile 累加；然后把编译后的 Triton kernel 对照 PTX 文本，读出 `.target` / `.reqntid` / `%ctaid.x` / `%tid.x` / `ld.global.v4.b32` 这些执行模型信号。读完后应能解释 thread coarsening 在 PTX 层体现为几条向量化指令。最后给出一条工具选择链：PyTorch builtin → `torch.compile` → Triton → CUDA C++ / PTX / SASS。

理想路径是把 $A$ 和 $B$ 都放进 shared memory 后再计算 $C$，这样读次数能从 $O(MKN)$ 降到接近 $O(MK + KN)$。

访存账本可以严格推导。**朴素路径**：固定一个输出元素 $C[m, n]$，对每个 $k$ 读一次 $A[m, k]$ 与 $B[k, n]$，共 $K$ 次读；遍历所有 $(m, n)$ 共 $MN$ 个输出元素，总 reads = $M \cdot N \cdot K = O(MNK)$，writes = $MN$。每个 $A[m, k]$ 在 $N$ 个不同的 $n$ 处被读 $N$ 次，每个 $B[k, n]$ 在 $M$ 个不同的 $m$ 处被读 $M$ 次——大量重复访存。

**理想路径（全部进 shared memory）**：把 $A$ 和 $B$ 整体放进 shared memory，每个 $A[m, k]$ 只读一次，每个 $B[k, n]$ 只读一次。Reads = $MK + KN$，writes = $MN$，因此 arithmetic intensity 从 $O(1)$（朴素）提到 $O(N)$（理想），跨过 roofline 的 $I_{\text{ridge}}$ 边界。

现实中矩阵太大，shared memory 放不下完整矩阵，所以采用 **tiling**：一个 program 负责 $C$ 的一个 $(\mathrm{BLOCK\_M}, \mathrm{BLOCK\_N})$ 输出 tile，沿 $K$ 维逐段加载 $A$ tile 和 $B$ tile，累加 partial sum，最后写回 HBM。每个 tile 的 $A$ tile 在 $\mathrm{BLOCK\_N}$ 个输出列中复用，每个 $B$ tile 在 $\mathrm{BLOCK\_M}$ 个输出行中复用，reads ≈ $MK + KN$ 数量级，写 $MN$ 个输出 tile。这是 tiled matmul 优于朴素 matmul 的根本原因，也是 §6.1 末段算术强度提升的具体落点。

![图 6.5-1 GEMM tiling data reuse](images/6-5-1-gemm-tiling.png)

*图 6.5-1 GEMM tiling data reuse*

图 6.5-1 右侧橙色块是当前输出 tile；为计算它，kernel 沿 $K$ 维读取左侧 $A$ 的行 tile 和中间 $B$ 的列 tile。紫色块表示外层 tile 扫描，绿色块表示内层元素乘加。tile 越大，每次 HBM 读入后能服务更多乘加，算术强度越高；tile 太大会增加 shared memory 和 register 压力，降低可驻留 block 数。

Triton 版 matmul + ReLU 的 wrapper 先确定 $M$、$K$、$N$，再启动二维 grid。每个 program 对应 $C$ 的一个 $(\mathrm{BLOCK\_M}, \mathrm{BLOCK\_N})$ tile。

```python
def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return c
```

Kernel 内部的核心循环沿 $K$ 维移动。`tl.dot(a, b)` 负责 tile 级矩阵乘法；`tl.maximum(acc, 0.0)` 在写回前融合 ReLU，避免先写未激活的 $C$，再启动另一个 elementwise kernel 读回来做激活。

```python
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

acc = tl.maximum(acc, 0.0)
tl.store(c_ptrs, acc, mask=...)
```

Triton 代码最终会编译到 PTX（Parallel Thread Execution）。`triton_gelu-ptx.txt` 可以用来观察几个直接对应执行模型的信号：

| PTX 信号 | 含义 |
| --- | --- |
| `.target sm_100a` | 面向 Blackwell / B200 一类目标架构 |
| `.reqntid 128` | 一个 thread block 请求 128 个 thread |
| `%ctaid.x` | 当前 block / CTA 的 x 维 id |
| `%tid.x` | 当前 thread 在 block 内的 x 维 id |
| `ld.global.v4.b32` | 从 global memory 做 4 个 32-bit 值的向量化 load |
| `st.global.v4.b32` | 向 global memory 做 4 个 32-bit 值的向量化 store |

*表 6.8 Triton GeLU PTX 中的执行模型信号*

这几个信号合起来能算出 thread coarsening 的倍数：`BLOCK_SIZE` 是 1024，而 `.reqntid 128` 说明一个 block 只有 128 个 thread，因此每个 thread 要处理 $1024 / 128 = 8$ 个元素。PTX 把这 8 个元素合并成 `ld.global.v4.b32`（每条 load / store 4 个 32-bit 值）形式的向量化指令：编译器把相邻元素合并进同一个 thread 的向量 load / store，减少指令数与调度开销，同时提高每个 thread 的工作量。精确的指令条数随编译器版本与 BLOCK_SIZE 不同而变化，需要对照实际 PTX 输出核证。

PTX 还不是硬件行为的全部：warp 调度、具体 SM 分配和许多微架构细节要继续下探到 profiler 或 SASS 才能确认。

本章的工具选择可以压缩成一条工程判断链：PyTorch builtin 优先；逐元素链条先试 `torch.compile`；需要改变计算组织、显式控制 tile、mask、访存和片上复用时使用 Triton；只有在需要更细粒度硬件控制或复用已有 CUDA 生态时，才继续下探到 CUDA C++、PTX 或 SASS。这条判断链来自维护成本与可读性的工程权衡，不是从公理推导得到的——经验规则，依赖硬件版本与团队熟悉度。

## 本章总结与下章衔接

读完本章后应能做到：写一个可复用的 benchmark 和 `torch.profiler` 排查流程，在 Triton 里覆盖 elementwise / reduction / row-overflow / matmul tiling 四类 block 级 kernel，并在 PTX 文本中读出执行模型与 thread coarsening 信号。第 5 章 给出硬件数量级与优化原则，本章把同一套判断落到具体 kernel 和工具选择链上。

下章进入 [第 7 章 §7.2 通信编程模型](../chapter7/chapter7_分布式训练.md)：单卡账本成立之后，把同一组账本扩展到跨卡——collective 语义、NCCL / torch.distributed 的实际接口、ZeRO / FSDP 的状态分片，以及 TP / PP / SP / CP / EP 在混合并行中的组合（EP 在 [第 7 章 §7.9 SP / CP / EP：Activation 与长上下文 / MoE 维度的并行](../chapter7/chapter7_分布式训练.md) 与 [第 4 章 §4.4 MoE 与深度学习](../chapter4/chapter4_混合专家模型.md) 章节交叉）。

## 思考

- 用 `torch.cuda.Event` 测一段 forward + backward 端到端时间，瓶颈是 kernel launch 还是 kernel 执行？
- 在 `torch.compile` 之后用 `nvprof` / `nsys` 看哪些 kernel 实际被 fuse 了，哪些仍是单点？
- 在 Triton kernel 里把 BLOCK_SIZE 从 64 改到 128、256、512，shared memory 占用和 occupancy 怎么变化？
- 同一个 reduction 在 `torch.sum` / 手写 Triton / `cub::BlockReduce` 三个实现下的 register pressure 和 SM 占用对比。

## 参考文献

- Philippe Tillet, H. T. Kung, David Cox. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)，MAPL 2019
- [Triton 官方文档](https://triton-lang.org/)
- [PyTorch `torch.compile` 文档](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [Stanford CS336 Lecture 6 课件与代码](https://github.com/stanford-cs336/lectures)

## 来源与更新记录

- 来源：本章以 CUTLASS 3.x 源码、Triton 文档、PTX ISA 与 NVIDIA H100/B200 datasheet 为主；`triton_gelu-ptx.txt` 等 PTX 一手输出物仅作可访问示例引用。
- 硬件规格：[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)（register file 64K 32-bit registers/SM、max 255 registers/thread、portable cluster size 8 与 B200 的非 portable cluster size 16；shared memory 容量 256 KB/SM，max 228 KB/SM carveout，max 227 KB/CTA），查阅日期 2026-09-05，状态：官方。
- Tensor Memory：[CUDA PTX ISA — Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory) 与 Colfax Research 的 Blackwell TMEM GEMM 教程（256 KB/SM、128 lane × 512 column、`tcgen05.alloc` / `tcgen05.dealloc` 由单个 warp 发起），查阅日期 2026-09-05，状态：官方 + 社区实现说明。
- Wave / tile quantization：[NVIDIA Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)（$256 \times 128$ tile、A100 108 SM 的一波 tile 数），查阅日期 2026-09-05，状态：官方。
- Fused softmax 访存账本：[Triton fused softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)（朴素实现总访存 $8MN + 4M$、理想 $2MN$、理论加速约 4 倍），查阅日期 2026-09-05，状态：官方文档。
- 表 6.2 的 register bandwidth 为按 SM 数、时钟与寄存器端口宽度换算的量级估计，其余三级带宽取自各代 datasheet 公布值。
- 参考：[`triton_gelu-ptx.txt`](https://github.com/stanford-cs336/lectures/blob/main/var/triton_gelu-ptx.txt)；Triton 文档；PyTorch `torch.compile` 文档。
