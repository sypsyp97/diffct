# 候选分支的结果验证

本文记录两轮 GPU 验证：2026-10-08 在 Leonardo Booster 上对当前代码的单节点、
跨节点验证，以及 2026-09-29 在 Alex 上对初版 `37a12d9` 的单节点验证。

## 当前代码：Leonardo Booster(2026-10-08)

环境：A100-SXM-64GB,驱动 535,Torch `2.10.0+cu126`,numba-cuda 0.30.4。

**正确性**

- 完整测试 **229 通过**(两张 A100)。
- 8 rank 跨节点 NCCL 检查(两个节点，每节点 4 卡)通过：平行束、扇束、螺旋锥束，
  视角分配不均和空 rank 都在内。
- Siddon 射线积分与独立 CPU float64 线段长度对比(源点到探测器像素这一段):
  - 源点距体中心 1e2 到 1e6 个体素，扇束最大相对误差 ≤ 1.1e-6;探测器远至
    1e6 个体素时 ≤ 9.8e-7;锥束最大绝对误差 ≤ 2.6e-5。
  - 源点在体中心、探测器平面穿过体数据时，结果等于源点到探测器那一段。
  - `voxel_spacing=0.5` 时投影值等于物理长度线积分。
- 平行束 FBP 往返(投影、斜坡滤波、FBP):重建与原图的质心偏差 0.006 个体素，
  内部幅值 0.999;`voxel_spacing` 取 1 和 0.5 结果相同。
- 几何梯度：与独立 float64 CPU 精确 Siddon 模型的中心差分(步长 1e-6)对比，
  三种束型所有几何分量的相对误差在 1e-9 到 4e-6 之间(个别小量 8e-5)。
  反投影的几何梯度与正投影一致(相对 7e-7);4 卡和两节点 8 卡的几何梯度与单卡
  一致(相对 2e-6)。
- 二阶导：体数据的 Hessian-向量积与 `AᵀA v` 一致(相对 2e-7);对几何求二阶导按设计报错。
- 与初版 `37a12d9` 同卡对比，单卡正投影 256³ 为 137.6 ms 对 135.8 ms,
  反投影 386.9 ms 对 387.8 ms。

**实测加速**:螺旋锥束、1024 个视角，体素和探测器间距为 1;128³ 用
`(192, 128)` 探测器,256³ 用 `(384, 256)`。完整梯度迭代,9 次中位数，
分布式耗时取最慢 rank。多卡结果与单卡在 `rtol=5e-4, atol=5e-5` 内一致。

| 配置 | 体数据 | 单卡 | 多卡 | 加速比 |
| --- | --- | ---: | ---: | ---: |
| 单节点 4 卡 | 128³ | 68.15 ms | 18.20 ms | **3.74×** |
| 单节点 4 卡 | 256³ | 526.68 ms | 136.70 ms | **3.85×** |
| 两节点 8 卡 | 128³ | 68.13 ms | 10.90 ms | **6.25×** |
| 两节点 8 卡 | 256³ | 526.64 ms | 80.95 ms | **6.51×** |

## 初版代码：Alex(2026-09-29)


初版基于 `dev` 的 `cb516cf`，提供 `Projector` 任意逐视角几何接口。
候选分支为 `codex/arbitrary-trajectory-multigpu`，供集群验证；
GitHub 默认分支和 PyPI 发布版本保持原状。

### 正确性

- Windows RTX 4070 SUPER：完整测试 **100 通过、6 跳过**。跳过的是需要两张 GPU 的测试。
- Alex 作业 **4407851**，节点 `a0905`，两张 A100-SXM4-40GB：完整测试 **106 通过**。
- 同一作业的真实 NCCL 两进程检查：每个 rank 的 6 个案例全部通过，覆盖平行束、扇束、螺旋锥束、视角分配不均、空 rank 和两种反向传播。
- 正投影以独立 CPU float64 均匀长方体射线交长为参考，验证三种束型。128³ 的抽样射线最大绝对误差为 `0.001607`，积分量级约 `182`；检查容差为 `rtol=1e-4, atol=2e-4`。
- 多卡正投影、匹配反投影和图像梯度通过逐元素比较，容差为 `rtol=5e-4, atol=5e-5`。浮点累加顺序不同，因此不要求逐位相同。
- 合成螺旋轨迹重建的残差由 `26922.10` 降至 `661.30`，10 次迭代后 MSE 为 `0.00609735`，与单卡验证一致。
- 固定几何的调用者修改保护、原始输入梯度 dtype、非默认 CUDA stream、跨 stream 缓存复用和调用者自有进程组均已检查。CPU/CUDA 混合几何校验问题已有先失败后通过的回归测试。

### 实测加速

同一 Alex 作业、同型号 GPU，512 个螺旋视角，体素和探测器间距均为 1。
64³ 使用 `(96, 64)` 探测器，128³ 使用 `(192, 128)` 探测器。
各模式单独配对测量单卡基准；预热后取 **9 次中位数**。
表中是完整 `0.5 * ||A x||²` 正投影及图像梯度迭代，包含每次调用的数据传输、
结果合并、autograd 和 NCCL 归约。分布式耗时取最慢 rank。

| 两卡模式 | 体数据 | 单卡 | 两卡 | 加速比 |
| --- | --- | ---: | ---: | ---: |
| NCCL，两进程 | 64³ | 6.558 ms | 4.499 ms | **1.46×** |
| NCCL，两进程 | 128³ | 33.420 ms | 17.392 ms | **1.92×** |
| 单进程 `devices=[0, 1]` | 64³ | 5.617 ms | 6.739 ms | **0.83×，未加速** |
| 单进程 `devices=[0, 1]` | 128³ | 33.440 ms | 22.851 ms | **1.46×** |

对实测的 64³ 任务，推荐 NCCL 两进程模式。单进程多卡在该规模的完整迭代没有
加速；默认本地加速检查保留这个失败结果，不降低门槛。作业 4407851 的 Slurm
退出码因此是 1，数值测试和 NCCL 加速检查均通过。

较小任务不能假定增加 GPU 就会加速。每张 GPU 保存完整体数据，当前设计不合并
多卡显存。应按实际体数据、探测器和视角数重新运行基准。

```bash
python -m pytest tests/ -q
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/distributed_projector_check.py --require-cuda \
    --expected-world-size=2 --output nccl-check.json
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/benchmark_projector.py --repeats 9 --output nccl-benchmark.json
python examples/benchmark_projector.py --devices 0 1 --repeats 9 \
    --output local-benchmark.json
```

### 环境与记录

Torch `2.13.0+cu130`、CUDA 13.0、驱动 `610.57.04`。当时验证的
`diffct/operators.py` SHA-256 为
`E8BFDAE28B198DA9D951E4D864967A8710CF3BF51E891CB326FFF620189118F5`。
原始日志、逐 rank JSON 和计时样本保存在本地 `.validation/`。

## 复现命令

```bash
python -m pytest tests/ -q
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/distributed_projector_check.py --require-cuda \
    --expected-world-size=2 --output nccl-check.json
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/benchmark_projector.py --repeats 9 --output nccl-benchmark.json
```

跨节点命令见 [DISTRIBUTED.md](DISTRIBUTED.md)。单节点 NCCL 结果不能代替
跨节点实测。
