# diffct：任意轨迹的可微 CT 算子

此候选基于 `dev`，分支为 `codex/arbitrary-trajectory-multigpu`。
GitHub 默认分支和 PyPI 发布版本
尚未修改。投影和匹配反投影默认使用逐视角几何，可用于圆轨迹、螺旋轨迹
以及用户提供的标定轨迹。

## 简单 API

```python
import torch
from diffct import Projector, spiral_trajectory_3d

trajectory = spiral_trajectory_3d(
    60, sid=100.0, sdd=160.0, z_range=12.0, n_turns=1.0, device="cpu"
)
operator = Projector(trajectory, volume_shape=(32, 32, 32),
                     detector_shape=(48, 40), detector_spacing=(1.0, 1.0))
volume = torch.ones(operator.volume_shape, device="cuda", requires_grad=True)
sinogram = operator.project(volume)
adjoint = operator.backproject(sinogram)
sinogram.square().sum().backward()
```

几何设置一次即可重复使用。二维扇束使用 `beam="fan"`，平行束使用
`beam="parallel"`；图像形状为 `(height, width)`，探测器大小为整数。
三维体数据形状为 `(depth, height, width)`，投影为 `(views, u, v)`，
探测器大小和间距均按 `(u, v)` 排列。体素间距目前是各向同性标量。

可直接传入几何辅助函数返回的 tuple，也可提供自己的源位置、探测器中心
和单位轴向量。几何固定，图像与投影数据支持 autograd；几何参数梯度不在
此接口的支持范围内。

`backproject()` 是匹配伴随算子，用于迭代重建，并不直接生成 FDK 重建。
任意轨迹可以进行投影和迭代重建，解析 FBP/FDK 仍需满足相应扫描条件。

## 多 GPU 与跨节点

单进程多 GPU 增加 `devices=["cuda:0", "cuda:1"]` 即可。算子按视角
分工，正投影保持视角顺序，反投影累加各卡结果，输出回到输入设备。
每张卡保留完整体数据，不能借此装入单卡无法容纳的体数据。

跨进程、跨节点使用 PyTorch 的 NCCL 进程组和 `distributed=True`。
各 rank 返回自己的投影视角，`operator.view_slice` 用于切分实测数据。
反投影和图像梯度跨 rank 求和。所有 rank 必须调用相同的通信和反向传播
序列，即使本 rank 没有视角。

投影域的局部损失使用求和；对每个 rank 都有的完整反投影结果计算损失时，
损失除以 `operator.world_size`。不要叠加 DDP 的图像梯度归约。

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/distributed_reconstruction.py
```

详见 [跨节点运行说明](docs/DISTRIBUTED.md)。支持在 Alex 或 tinygpu 的
同一个 Slurm allocation 内跨节点运行；两个独立集群间通信还需要独立的
网络配置和验证。

结果与加速验证使用 `examples/benchmark_projector.py`：先用 CPU 解析参考
核对单卡射线积分，再比较多卡投影、反投影和图像梯度。它会实测包含传输、
合并和通信的完整梯度迭代时间，报告加速比；默认比较 64³ 和 128³ 两个体
数据规模。小任务可能受调度开销影响，应按实际扫描规模验证。

```bash
python examples/benchmark_projector.py --devices 0 1 --output gpu-benchmark.json
```

Alex 两张 A100 的完整迭代实测：NCCL 两进程在 64³、128³ 上分别加速
1.46×、1.92×；单进程多卡在 128³ 上加速 1.46×，64³ 上未加速。
详细数据与验证边界见 [结果报告](docs/VALIDATION.md)。

## 安装与迁移

从本地工作区执行 `pip install -e .`。CUDA/PyTorch/Numba 的环境设置见
[英文说明](README.md)。NumPy 当前限制为 `<2.5`，以避开 Numba CUDA
的已复现兼容问题。

从旧 `main` 迁移时，注意探测器半个 bin 的坐标约定变化。旧 `main` 的
SF 后端尚未移植到任意轨迹实现。底层 Function API 继续保留 `dev` 的形式。
详见 [迁移说明](docs/MIGRATION.md)。

```bash
python -m pytest tests/ -q
```

许可：[Apache 2.0](LICENSE)。引用方式见英文 README。
