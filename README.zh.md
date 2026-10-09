<h1 align="center">diffct</h1>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

<p align="center">
  面向 CT 的可微 CUDA 投影算子：任意轨迹、多卡、多节点、几何梯度。
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.14999333"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square" alt="DOI"></a>
  <a href="https://pypi.org/project/diffct/"><img src="https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="docs/source/trajectories.rst"><img src="https://img.shields.io/badge/docs-branch-brightgreen.svg?style=flat-square" alt="Branch documentation"></a>
  <a href="https://github.com/sypsyp97/diffct/actions"><img src="https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=CI&style=flat-square" alt="CI/CD"></a>
  <a href="https://deepwiki.com/sypsyp97/diffct"><img src="docs/assets/deepwiki-badge.svg" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  <img src="docs/assets/diffct_intro.gif" width="100%" alt="diffct 动图：投影、正弦图、轨迹、多卡切分、重建和几何标定">
</p>

<p align="center">
  <a href="docs/assets/diffct_intro.mp4">完整视频（MP4）</a> ·
  <a href="https://www.preprints.org/manuscript/202605.1446/v1">技术报告</a> ·
  <a href="https://doi.org/10.20944/preprints202605.1446.v1">DOI</a>
</p>

<p align="center">
  <a href="#安装">安装</a> ·
  <a href="#快速上手">快速上手</a> ·
  <a href="#多卡与多节点">多卡</a> ·
  <a href="#性能">性能</a> ·
  <a href="#示例图">示例图</a> ·
  <a href="#示例">示例</a> ·
  <a href="README.md#citation">引用</a>
</p>

> 本文对应候选分支 `codex/arbitrary-trajectory-multigpu`。GitHub 默认分支和
> PyPI 版本还没有 `Projector` 接口。请先看[本分支指南](docs/source/trajectories.rst)，
> 从旧版 API 迁移前请读[迁移说明](docs/MIGRATION.md)。
>
> Apple/MLX 移植版由 [Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider) 维护，
> 见 [DiffCT-MLX](https://github.com/Linda-SophieSchneider/DiffCT-MLX)。

## 功能与边界

| 方面 | 本分支支持 | 重要限制 |
|---|---|---|
| 采集几何 | 二维平行束、扇束及三维锥束；逐视角源点与探测器几何 | 平面探测器，方向轴须为单位向量；任意轨迹不支持 `sf`、`sf_tr`、`sf_tt` 后端 |
| 自动微分 | 体数据、正弦图的一阶与二阶导；几何的一阶导 | 几何二阶导会报错；几何有效性只在构造时检查 |
| 执行方式 | CUDA 单卡、单进程多卡、跨节点分布式 | 内核和输出均为 float32；几何可放在 CPU，投影计算必须使用 CUDA |
| 重建 | 分片常数 Siddon 模型的匹配伴随，以及独立的 FBP/FDK 辅助函数 | `backproject()` 不是逆算子；FDK 是近似方法，不因支持任意轨迹而成为精确重建 |
| 内存 | 按视角分配到 GPU 或 rank | 每张参与计算的 GPU 都需要完整体数据；单进程多卡还会在输入设备汇集完整正弦图 |

实测结果及其适用范围见 [验证记录](docs/VALIDATION.md)，不代表任意扫描或硬件的保证。

## 安装

需要 CUDA GPU 和 PyTorch。请先按你的 CUDA 版本安装 PyTorch。

```bash
git clone https://github.com/sypsyp97/diffct.git
cd diffct && git checkout codex/arbitrary-trajectory-multigpu
pip install "numpy<2.5" "numba-cuda[cu12]"   # CUDA 13 用 [cu13]；先按 CUDA 版本安装 PyTorch
pip install -e .
python examples/quickstart.py               # 冒烟测试：每种射束打印伴随误差约 1e-8
```

<details>
<summary>CUDA 版本说明</summary>

- Numba CUDA 导入 `numpy.row_stack`，NumPy 2.5 已删除该函数，所以保持 `numpy<2.5`。
- NVVM 和 NVJitLink 要与 PyTorch 加载的 CUDA 库兼容。较新的 NVVM 配较旧的
  NVJitLink 时，内核编译会失败。
- 已测试的 CUDA 13 组合：
  `pip install "numpy<2.5" "numba-cuda[cu13]" "cuda-toolkit[cccl,cudart,nvrtc,nvvm]==13.0.2" "nvidia-nvjitlink<13.1"`。
- 已测试的 CUDA 12 组合：PyTorch 2.10（cu126）加 `numba-cuda[cu12]` 0.30.4，驱动为 NVIDIA 535。

</details>

## 快速上手

先创建一次采集几何，再调用 `project()` 和 `backproject()`。

```python
import torch
from diffct import Projector, spiral_trajectory_3d

trajectory = spiral_trajectory_3d(360, sid=320.0, sdd=512.0, z_range=40.0, n_turns=1.0, device="cpu")
A = Projector(trajectory, volume_shape=(128, 128, 128), detector_shape=(384, 256), detector_spacing=0.8)

volume = torch.rand(A.volume_shape, device="cuda")
sinogram = A.project(volume)        # (360, 384, 256) 线积分
adjoint = A.backproject(sinogram)   # 匹配的伴随算子 A^T

x = torch.zeros_like(volume, requires_grad=True)
loss = 0.5 * (A.project(x) - sinogram).square().sum()
loss.backward()                     # x.grad = A^T (A x - y)
```

### 自定义或标定轨迹

标准扫描可用轨迹生成器；已有标定结果时，直接传入逐视角的源点、探测器中心和方向轴张量，
无需拟合成圆轨迹。以下是两个锥束视角的示例：

```python
calibrated = tuple(torch.tensor(rows, dtype=torch.float32) for rows in (
    [[-320.0, 0.0, -20.0], [0.0, -300.0, 25.0]],  # 源点 (x, y, z)
    [[192.0, 0.0, -20.0], [0.0, 212.0, 25.0]],    # 探测器中心 (x, y, z)
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],         # 探测器 u 方向单位向量
    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],         # 探测器 v 方向单位向量
))
A_custom = Projector(calibrated, (64, 96, 128), (192, 128),
                     beam="cone", voxel_spacing=1.0, detector_spacing=(0.8, 1.0))
y_custom = A_custom.project(torch.ones(A_custom.volume_shape, device="cuda"))
# y_custom.shape == (2, 192, 128)：(视角, detector_u, detector_v)
```

实际使用时，将每个分量的两行替换为完整扫描的标定数组，行顺序与测量数据一致。
几何采用世界坐标 `(x, y, z)`，体数据排列为 `(z, y, x)` / `(D, H, W)`。
位置与间距须使用同一长度单位；方向轴是单位向量，不含像元间距。
二维元组和居中约定见[几何与单位](#几何与单位)。
`custom_trajectory_3d` 则从源点路径自动推导朝向原点的探测器位姿；
若探测器位姿由独立标定得到，应直接传入元组。

## 多卡与多节点

按投影数据的存储方式选择执行模式：

| 模式 | 配置 | 投影数据归属 |
|---|---|---|
| 单卡 | 默认 `Projector(...)` | 完整正弦图位于输入 CUDA 设备 |
| 单进程多卡 | `devices=[0, 1, ...]` | 分卡计算视角，再将完整正弦图汇集到输入 CUDA 设备 |
| 每卡一个进程，单节点或多节点 | 初始化进程组后设置 `distributed=True` | 各 rank 只持有自己的视角，形状为 `A.projection_shape`，索引为 `A.view_slice` |

每张参与计算的 GPU 都需要完整体数据。单进程多卡还需要在调用方设备上容纳完整的输入或输出正弦图，
以及临时副本，不能将多卡显存合并使用。分布式模式下正弦图保持分片，反投影结果则求和后复制到各 rank。
加速比取决于工作量和通信开销。

**单进程多卡：**

```python
A = Projector(trajectory, (128, 128, 128), (384, 256), detector_spacing=0.8, devices=[0, 1, 2, 3])
```

**每卡一个进程，可跨一个或多个节点：**

```bash
torchrun --nproc-per-node=4 examples/iterative_reconstruction.py --trajectory helical      # 单节点
sbatch examples/slurm/multi_node.sbatch examples/iterative_reconstruction.py --trajectory helical   # 多节点
```

设置 `distributed=True` 时，每个 rank 都必须执行相同的 `project`、`backproject` 和反向调用，
包括没有分到视角的 rank。投影损失按各 rank 的本地视角计算，并采用 SUM 语义；算子会自动对图像和几何梯度跨 rank 求和。
对各 rank 复制持有的反投影结果计算损失时，须除以 `world_size`。不要在算子之外再叠加 DDP 梯度归约。
初始化、损失示例、Slurm 细节和跨节点数值检查见 [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md)。

## 几何梯度

给轨迹张量设置 `requires_grad=True`，即可得到源点、探测器中心和探测器轴的梯度。
该梯度是分片常数模型的精确导数。须在构造 `Projector` 前设置 `requires_grad=True`；
否则算子会保存几何快照，修改固定几何后需要重新构造算子。

```python
source, det_center, det_u, det_v = (t.clone() for t in trajectory)
source.requires_grad_()
A = Projector((source, det_center, det_u, det_v), (128, 128, 128), (384, 256), detector_spacing=0.8)
(A.project(volume) - sinogram).square().sum().backward()   # source.grad 形状为 (360, 3)
```

对几何求二阶导会报错。边界情况见[几何与单位](#几何与单位)。

## 性能

Leonardo Booster（NVIDIA A100 64 GB，PyTorch 2.10 cu126，numba-cuda 0.30.4）上每次迭代的耗时，
从 1 卡到单节点 4 卡、双节点 8 卡：

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/scaling_dark.png">
    <img src="docs/assets/scaling_light.png" width="100%" alt="1、4、8 卡上正投影、伴随和 CGLS 每次迭代的耗时，64³ 和 128³">
  </picture>
</p>

| 128³ 体数据 | 1 卡 | 4 卡，1 节点 | 8 卡，2 节点 |
|---|---:|---:|---:|
| 正投影 | 8.46 ms | 2.67 ms（3.2×） | 1.69 ms（5.0×） |
| 伴随（反投影） | 20.69 ms | 5.91 ms（3.5×） | 4.53 ms（4.6×） |
| CGLS 迭代 | 30.07 ms | 8.09 ms（3.7×） | 5.54 ms（5.4×） |

圆轨迹，360 个视角，探测器 (2n, 1.5n) 个像元，间距 1.25。
小问题的加速比较低。64³ 的一次 CGLS 迭代，1 卡为 4.75 ms，4 卡为 1.57 ms（3.0×），
8 卡为 1.56 ms（3.0×）。此时数据传输和内核启动占主要部分。

在单进程多卡上复现：

```bash
python examples/benchmark_projector.py --devices 0 1
```

torchrun 的启动方式见 [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md)。

## 示例图

**螺旋扫描，1% 噪声。** Shepp-Logan 体模，128³。PSNR：FDK 31.2 dB，CGLS（30 次迭代）35.8 dB，
SIRT（200 次迭代）32.7 dB，TV（200 次迭代）47.5 dB。

<p align="center">
  <img src="docs/assets/iterative_helical_noise.png" width="100%" alt="1% 噪声下螺旋扫描的重建结果">
</p>

```bash
python examples/iterative_reconstruction.py --trajectory helical --noise 0.01 --figure out.png
```

**解析重建。** Shepp-Logan 窗。PSNR：平行束 FBP 38.1 dB，扇束 FBP 34.9 dB，锥束 FDK 34.1 dB。

<p align="center">
  <img src="docs/assets/analytical.png" width="100%" alt="平行束 FBP、扇束 FBP 和锥束 FDK 的重建结果">
</p>

```bash
python examples/analytical_reconstruction.py --figure out.png
```

## 示例

| 文件 | 内容 | 启动方式 |
|---|---|---|
| `quickstart.py` | 平行束、扇束、锥束的 Projector 基础：正投影、反投影、伴随检查、图像和几何梯度 | 1 卡 |
| `analytical_reconstruction.py` | 平行束 FBP、扇束 FBP、锥束 FDK；`--window` | 1 卡 |
| `iterative_reconstruction.py` | 任意轨迹（`--trajectory circular/helical/saddle/sinusoidal`）；CGLS、SIRT、带 TV 正则的非负最小二乘（Adam，autograd）；FDK 基线；`--noise` | 1 卡，`--devices 0 1 2 3`，torchrun，Slurm 多节点 |
| `geometry_calibration.py` | 用几何梯度从投影中恢复每视角角度误差和探测器偏移 | 1 卡，`--devices`，torchrun，Slurm 多节点 |
| `benchmark_projector.py` | 单卡与多卡的正确性和速度 | 单进程 `--devices`，torchrun |
| `plot_trajectory.py` | 绘制轨迹生成器的输出 | CPU |
| `slurm/multi_node.sbatch` | 模板：每节点一个 torchrun 启动器；设置账号和分区 | Slurm |

启动方式、分布式损失的规则和全部实测结果见 [examples/README.md](examples/README.md)。

## 几何与单位

| 射束 | 轨迹元组，每行一个视角 | 体数据 | 正弦图 |
|---|---|---|---|
| `parallel` | `(ray_dir, det_origin, det_u)`，各分量为 `(views, 2)` | `(H, W)` | `(views, U)` |
| `fan` | `(src_pos, det_center, det_u)`，各分量为 `(views, 2)` | `(H, W)` | `(views, U)` |
| `cone` | `(src_pos, det_center, det_u, det_v)`，各分量为 `(views, 3)` | `(D, H, W)` | `(views, U, V)` |

- 可用 `diffct.geometry` 中的 `circular_*`、`spiral_*`、`sinusoidal_*`、`saddle_*`、
  `random_*`、`custom_*` 函数生成轨迹，也可直接传入标定张量。
- 几何采用世界坐标 `(x, y)` 或 `(x, y, z)`；体数据轴顺序为 `(y, x)` 或 `(z, y, x)`，不含 batch/channel 维度。
- 方向向量为单位向量。平行束中 `ray_dir` 与 `det_u` 正交，锥束中 `det_u` 与 `det_v` 正交。
  `detector_spacing` 在二维时为标量，锥束时为标量或 `(du, dv)`。
  锥束的 `detector_shape=(U, V)`，正弦图顺序固定为 `(views, U, V)`。
- 体数据以原点为中心。长度为 `N` 的轴上，第 `i` 个体素的中心位于
  `(i + 0.5 - N / 2) * voxel_spacing`。体素间距为各向同性的单个值。
- 探测器阵列居中。共 `N_det` 个像元时，第 `k` 个像元位于 `det_center` 沿 `det_u` 方向的
  `(k - (N_det - 1) / 2) * pitch` 处，平行束的参考点为 `det_origin`，与 `main` 分支的约定相同。
  锥束还需按 `det_v` 方向及其像元间距加上对应偏移。
- 投影值是几何长度单位下的线积分。扇束和锥束的射线从源点到探测器像元。
- 源点与探测器中心重合，或探测器侧对源点（边缘朝向源点）的视角，会被 `Projector` 拒绝。
- 内核使用 float32。扇束和锥束中，每个视角的源点或探测器中心须位于体数据中心 1e6 个体素以内。
  两者都须位于 1e15 个体素以内。几何可以在 CPU 或 CUDA 上，但传给算子的体数据和正弦图必须是浮点 CUDA 张量。
  射线位置的精度约为该较近距离的 6e-8 倍。

**梯度边界情况：**

- 射线恰好落在体素边或角上时，内核返回某一侧的导数，而中心差分会平均两侧，因此结果可能不同。
  对称几何容易遇到此情况；进行有限差分检查时，可用 `start_angle=0.1` 等方式轻微旋转轨迹。
- 几何有效性只在构造时检查。优化角度和偏移量，而非直接优化原始轴向量，以保持几何有效。

## 文档与引用

| 文档 | 内容 |
|---|---|
| [本分支指南](docs/source/trajectories.rst) | 轨迹元组、`Projector` 用法和分支适用范围 |
| [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md) | 执行与内存选择、分布式损失规则、Slurm 和跨节点检查 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 测试命令、验证细节和每项检查的边界 |
| [docs/MIGRATION.md](docs/MIGRATION.md) | 从仅支持圆轨迹的 API 迁移 |
| [docs/video/README.md](docs/video/README.md) | 动图视频的渲染方式 |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 |

在 CUDA 主机上运行测试：`python -m pytest tests/ -q`。

引用方式见 [英文 README 的 Citation](README.md#citation)。

许可：[Apache 2.0](LICENSE)。
