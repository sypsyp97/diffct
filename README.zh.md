# diffct：任意轨迹的可微 CT 算子

diffct 提供平行束、扇束和锥束的 CUDA 正投影与匹配反投影。每个视角有自己的
源位置、探测器位置和探测器轴，因此圆轨迹、螺旋轨迹和标定轨迹走同一套代码。
算子是 PyTorch autograd 函数，可在单卡、多卡和多节点上运行。

> 本文对应候选分支 `codex/arbitrary-trajectory-multigpu`。GitHub 默认分支和
> PyPI 版本还没有 `Projector` 接口。从旧 `main` 迁移前请读
> [迁移说明](docs/MIGRATION.md)。安装步骤和 CUDA 版本说明见
> [英文 README](README.md#installation)。

## 快速上手

```python
import torch
from diffct import Projector, spiral_trajectory_3d

trajectory = spiral_trajectory_3d(
    60, sid=100.0, sdd=160.0, z_range=12.0, n_turns=1.0, device="cpu"
)
operator = Projector(trajectory, volume_shape=(32, 32, 32),
                     detector_shape=(48, 40), detector_spacing=(1.0, 1.0))
volume = torch.ones(operator.volume_shape, device="cuda", requires_grad=True)
sinogram = operator.project(volume)        # (views, u, v)
adjoint = operator.backproject(sinogram)   # 匹配伴随，不是 FDK 重建
sinogram.square().sum().backward()
```

二维用 `beam="fan"` 或 `beam="parallel"`，图像形状 `(height, width)`，探测器
大小为整数。

## 几何与单位

- 轨迹是逐视角张量：平行束 `(ray_dir, det_origin, det_u)`，扇束
  `(src_pos, det_center, det_u)`，锥束 `(src_pos, det_center, det_u, det_v)`。
  可用 `diffct.geometry` 的辅助函数生成，也可传入标定结果。
- 方向向量为单位向量。平行束 `ray_dir` 与 `det_u` 正交，锥束 `det_u` 与
  `det_v` 正交。
- 体数据以原点为中心。长度为 `N` 的轴上，第 `i` 个体素中心在
  `(i + 0.5 - N / 2) * voxel_spacing`。体素间距为各向同性标量。
- 探测器第 `k` 个像素位于 `det_center + (k - N_det / 2) * pitch * det_u`。
- 投影值是几何长度单位下的线积分。扇束和锥束只积分源点到探测器像素这一段。
- 源点与探测器中心重合、或探测器侧对源点的视角会被拒绝。
- 内核用 float32 计算。扇束和锥束的每个视角里，源点或探测器中心至少有一个须在
  体中心 1e6 个体素以内；射线位置精度约为该较近距离的 6e-8 倍。
- 梯度传到体数据和投影数据；几何参数不求梯度。

## 多卡与跨节点

算子按视角分工，每张卡保存完整体数据。多卡能加速，但不能放下单卡放不下的
体数据。

- 单进程多卡：加 `devices=["cuda:0", "cuda:1"]`。
- 每卡一个进程（可跨节点）：初始化 NCCL 进程组，加 `distributed=True`。
  `project()` 只返回本 rank 的视角，`operator.view_slice` 用来切实测数据；
  `backproject()` 和图像梯度跨 rank 求和。
- 投影域损失按 rank 求和；对各 rank 都有的完整反投影计算损失时，除以
  `operator.world_size`。不要再叠加 DDP 梯度归约。
- 所有 rank 必须做相同的 `project`、`backproject` 和反向调用，没有视角的 rank
  也一样，否则其他 rank 会阻塞。

Slurm 启动方式和跨节点数值检查见 [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md)。

## 验证

```bash
python -m pytest tests/ -q
```

Leonardo Booster(A100-SXM-64GB,NCCL,每卡一个进程)实测：螺旋锥束、1024 个
视角，一次梯度迭代(正投影加图像梯度)，取 9 次中位数。多卡结果与单卡在
`rtol=5e-4` 内一致。

| GPU 数 | 128³ | 256³ |
|---|---:|---:|
| 1 | 68.0 ms | 526.0 ms |
| 4,单节点 | 18.2 ms(3.74×) | 136.7 ms(3.85×) |
| 8,两节点 | 10.9 ms(6.23×) | 80.9 ms(6.50×) |

小任务受数据传输和启动开销影响，加速更少。请用
`examples/benchmark_projector.py` 按实际规模测量。

详细数据和每项检查的边界见 [docs/VALIDATION.md](docs/VALIDATION.md)。

许可：[Apache 2.0](LICENSE)。引用方式见 [英文 README](README.md#citation)。
