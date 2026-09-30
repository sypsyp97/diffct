# 候选分支的结果验证（2026-09-29）

此候选基于 `dev` 的 `cb516cf`，提供 `Projector` 任意逐视角几何接口。
候选分支为 `codex/arbitrary-trajectory-multigpu`，供集群验证；
GitHub 默认分支和 PyPI 发布版本保持原状。

## 正确性

- Windows RTX 4070 SUPER：完整测试 **100 通过、6 跳过**。跳过的是需要两张 GPU 的测试。
- Alex 作业 **4407851**，节点 `a0905`，两张 A100-SXM4-40GB：完整测试 **106 通过**。
- 同一作业的真实 NCCL 两进程检查：每个 rank 的 6 个案例全部通过，覆盖平行束、扇束、螺旋锥束、视角分配不均、空 rank 和两种反向传播。
- 正投影以独立 CPU float64 均匀长方体射线交长为参考，验证三种束型。128³ 的抽样射线最大绝对误差为 `0.001607`，积分量级约 `182`；检查容差为 `rtol=1e-4, atol=2e-4`。
- 多卡正投影、匹配反投影和图像梯度通过逐元素比较，容差为 `rtol=5e-4, atol=5e-5`。浮点累加顺序不同，因此不要求逐位相同。
- 合成螺旋轨迹重建的残差由 `26922.10` 降至 `661.30`，10 次迭代后 MSE 为 `0.00609735`，与单卡验证一致。
- 固定几何的调用者修改保护、原始输入梯度 dtype、非默认 CUDA stream、跨 stream 缓存复用和调用者自有进程组均已检查。CPU/CUDA 混合几何校验问题已有先失败后通过的回归测试。

## 实测加速

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

## 跨节点状态与证据

真实双节点 GPU 验证尚未完成。Alex 的 FAU 资源配额不允许多节点作业，
需要单独的 NHR 项目。后续应在已获多节点权限的集群上实测。
[双节点运行说明](DISTRIBUTED.md) 同时覆盖不同主机检查、数值一致性和完整
迭代加速。单节点的 NCCL 结果不能代替双节点实测。

验证环境：Torch `2.13.0+cu130`、CUDA 13.0、驱动 `610.57.04`。
经验证的 `diffct/operators.py` SHA-256：
`E8BFDAE28B198DA9D951E4D864967A8710CF3BF51E891CB326FFF620189118F5`。
源码快照和原始日志、逐 rank JSON、9 次计时样本保存在本地 `.validation/`，
其中 `nccl-benchmark-4407851.json` 与 `local-benchmark-4407851.json` 对应上表。
工作区已构建包含同一算子源码的 wheel，供本地安装验证。
