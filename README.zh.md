<h1 align="center">diffct</h1>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

<p align="center">
  面向 CT 的可微 CUDA 投影算子：任意轨迹、多卡、多节点、几何梯度。
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.14999333"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square" alt="DOI"></a>
  <a href="https://pypi.org/project/diffct/"><img src="https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://sypsyp97.github.io/diffct/"><img src="https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=docs&style=flat-square" alt="Documentation"></a>
  <a href="https://github.com/sypsyp97/diffct/actions"><img src="https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/ci.yml?branch=main&label=CI&style=flat-square" alt="CI/CD"></a>
  <a href="https://deepwiki.com/sypsyp97/diffct"><img src="docs/assets/deepwiki-badge.svg" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/sypsyp97/diffct/edf7c52227ec2cc14c988918faa7313f8ee0fe03/docs/assets/diffct_intro.gif" width="100%" alt="diffct 动图：投影、正弦图、轨迹、多卡切分、重建和几何标定">
</p>

<p align="center">
  <a href="https://github.com/sypsyp97/diffct/blob/edf7c52227ec2cc14c988918faa7313f8ee0fe03/docs/assets/diffct_intro.mp4">完整视频（MP4）</a> ·
  <a href="https://www.preprints.org/manuscript/202605.1446/v1">技术报告</a> ·
  <a href="https://doi.org/10.20944/preprints202605.1446.v1">DOI</a>
</p>

<p align="center">
  <a href="#安装">安装</a> ·
  <a href="#快速上手">快速上手</a> ·
  <a href="#性能">性能</a> ·
  <a href="#示例图">示例图</a> ·
  <a href="#示例">示例</a> ·
  <a href="#引用">引用</a>
</p>

> **安装**请带 CUDA extra，在 PyTorch 之后运行 `pip install "diffct[cu12]"` 或 `pip install "diffct[cu13]"`。只运行 `pip install diffct` 需要系统已装 CUDA Toolkit。
>
> **2.0 版**用逐视角轨迹取代了 1.x 的圆轨迹接口。为 1.x 写的代码需要修改，见[迁移说明](docs/MIGRATION.md)。
> Apple Silicon 请使用 [Linda-Sophie Schneider 的 MLX 移植版](https://github.com/Linda-SophieSchneider/DiffCT-MLX)。

## 为什么使用 diffct

据我们所知，diffct 是唯一同时具备以下四点的开源 GPU CT 库：任意逐视角轨迹、对体数据的自动求导、对采集几何的一阶梯度，以及内置的多卡和多节点执行。我们在 2026 年 10 月与 LEAP、TIGRE、ASTRA/tomosipo、DiffDRR 等库做了对比。

- **任意轨迹。** 每个视角有各自的源点、探测器中心和探测器轴；圆轨迹、螺旋、鞍形、正弦、随机和标定扫描使用同一套代码。`requires_grad=True` 的轨迹张量可获得用于标定的几何梯度。
- **曲面探测器。** 用 `detector_surface(u, v)` 定义每个像素的位置，原生支持弧形、柱面及其他参数化曲面；投影与匹配反投影支持曲面参数的一阶梯度。见[曲面接口](docs/REFERENCE.md#parameterized-detector-surfaces)（英文）。
- **匹配的算子。** `project()` 与 `backproject()` 对分片常数 Siddon 模型构成精确的伴随对；两者都支持 PyTorch 自动微分，包括体数据与正弦图的梯度和 Hessian 向量积。
- **大体积。** CPU 体数据和正弦图默认按空间块、视角批次送入 CUDA；完整数组和数据梯度留在 CPU。块大小按可用显存选择，也可手动覆盖。见[分块执行](docs/REFERENCE.md#chunked-execution)（英文）。
- **多卡与多节点。** 单进程使用 `devices=[0, 1, 2, 3]`，或每卡一个进程并用 torchrun 与 NCCL。视角分片；CPU 体数据留在主机内存，各 GPU 计算分块，分布式各 rank 保留完整体数据副本。加速比取决于工作负载和通信开销。
- **解析辅助函数。** `diffct.analytical` 提供斜坡滤波器（ram-lak、shepp-logan、cosine、hamming、hann）、扇束、锥束和 Parker 权重，以及 FBP 与 FDK 反投影。

功能边界、限制和等中心规则见 [docs/REFERENCE.md](docs/REFERENCE.md#capabilities-and-limits)（英文）。

## 安装

需要 NVIDIA GPU。先按你的 CUDA 版本安装 PyTorch，再按同一 CUDA 大版本安装 diffct：

```bash
pip install "diffct[cu12]"    # PyTorch 为 CUDA 12 构建
pip install "diffct[cu13]"    # PyTorch 为 CUDA 13 构建
```

extra 会装上 Numba CUDA 编译内核所需的 CUDA 编译库。只运行 `pip install diffct` 不会装这些库，仅适用于系统已装 CUDA Toolkit 的情况。

从源码安装（含示例）：

```bash
git clone https://github.com/sypsyp97/diffct.git
cd diffct && pip install -e ".[cu12]"
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

**自定义或标定轨迹。** 可直接传入标定得到的逐视角源点、探测器中心和方向轴张量，无需拟合圆轨迹。详见 [REFERENCE](docs/REFERENCE.md#custom-or-calibrated-trajectories)（英文）。

**几何梯度。** 构造 `Projector` 前，将轨迹张量设为 `requires_grad=True`。详见 [REFERENCE](docs/REFERENCE.md#geometry-gradients)（英文）。

**多卡与多节点。** 单进程使用 `devices=[0, 1, 2, 3]`；每卡一个进程并用 torchrun，可跨一个或多个节点。详见 [REFERENCE](docs/REFERENCE.md#several-gpus-and-nodes)（英文）和 [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md)。

```python
A = Projector(trajectory, (128, 128, 128), (384, 256), detector_spacing=0.8, devices=[0, 1, 2, 3])
```

**体积超过显存容量。** 把数据放在 CPU，同一套接口会自动分块计算；CUDA
输入在估计的工作集能装下时保留完整体积路径。无需开启分块开关：

```python
A = Projector(trajectory, (128, 128, 128), (384, 256), detector_spacing=0.8)
volume = torch.ones(A.volume_shape, device="cpu", requires_grad=True)
sinogram = A.project(volume)                 # CPU 输出，CUDA 计算
sinogram.square().mean().backward()          # volume.grad 在 CPU
# 可选手动覆盖：volume_chunk_shape=(32, 64, 64), view_chunk_size=16
```

运行 `python examples/chunked_reconstruction.py` 可演示 CPU 数据上的 CGLS
并测量 CUDA 张量分配峰值。分块保持同一 Siddon 模型，但浮点求和顺序会变化；
更小的块可能增加传输和内核启动开销。完整数组及迭代状态仍需足够的主机内存。

## 性能

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/scaling_dark.png">
    <img src="docs/assets/scaling_light.png" width="100%" alt="1、4、8 卡上正投影、伴随和 CGLS 每次迭代的耗时，64³ 和 128³">
  </picture>
</p>

| 128³ 体数据 | 1 卡 | 4 卡，1 节点 | 8 卡，2 节点 |
|---|---:|---:|---:|
| 正投影 | 8.45 ms | 2.66 ms（3.2×） | 1.68 ms（5.0×） |
| 伴随（反投影） | 20.66 ms | 5.89 ms（3.5×） | 4.62 ms（4.5×） |
| CGLS 迭代 | 30.08 ms | 8.08 ms（3.7×） | 5.59 ms（5.4×） |

圆轨迹，360 个视角，探测器 (2n, 1.5n) 个单元、pitch 1.25，A100 64 GB GPU。小体积加速比较低：64³ 时，一次 CGLS 迭代在 1 卡、4 卡和 8 卡上分别为 4.63 ms、1.57 ms 和 1.56 ms。原始数据见 [docs/assets/scaling.json](docs/assets/scaling.json)。

在本机比较单卡与多卡（脚本自带的默认扫描，不是上表的配置）：

```bash
python examples/benchmark_projector.py --devices 0 1
```

## 示例图

TV 指带全变分正则的迭代重建，Adam 是优化器。图中分别标注重建方法与优化器。

**实测核桃。** 240 个实测视角，圆形锥束，256³：FDK（Hann 窗）、SIRT（200 次迭代）、CGLS（20 次）、TV（300 次，权重 0.3）；轴向和冠状面中心切片。

<p align="center">
  <img src="docs/assets/walnut_measured.png?v=e0092c4" width="100%" alt="实测核桃：FDK、SIRT、CGLS 和 TV 重建结果">
</p>

```bash
python examples/walnut_reconstruction.py --figure out.png
```

**核桃螺旋扫描模拟。** 以 FDK 核桃体数据为真值，720 个视角，1% 噪声，256³。PSNR：FDK 26.67 dB，CGLS（30 次迭代）34.13 dB，SIRT（200 次）34.52 dB，TV（200 次，权重 1.0）37.77 dB。

<p align="center">
  <img src="docs/assets/walnut_helical.png?v=e0092c4" width="100%" alt="核桃螺旋扫描模拟：FDK、CGLS、SIRT 和 TV 重建结果">
</p>

```bash
python examples/walnut_reconstruction.py --algorithms --save-volume walnut256.npy
python examples/iterative_reconstruction.py --size 256 --views 720 --trajectory helical --noise 0.01 --phantom walnut256.npy --figure out.png
```

核桃数据：Meaney 2022，Zenodo 6986012，CC BY 4.0，见 [examples/data/NOTICE](examples/data/NOTICE)。

## 示例

`quickstart.py`、`curved_detector.py`、`chunked_reconstruction.py`、`analytical_reconstruction.py`、`iterative_reconstruction.py`、`walnut_reconstruction.py`、`geometry_calibration.py`、`benchmark_projector.py`、`plot_trajectory.py`，以及 Slurm 模板 `slurm/multi_node.sbatch`。启动方式和分布式损失规则见 [examples/README.md](examples/README.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [轨迹指南](docs/source/trajectories.rst) | 轨迹元组和 `Projector` 用法 |
| [docs/REFERENCE.md](docs/REFERENCE.md) | 功能边界、自定义轨迹、几何梯度、多卡细节、几何与单位（英文） |
| [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md) | 执行与内存选择、分布式损失规则、Slurm 和跨节点检查 |
| [docs/MIGRATION.md](docs/MIGRATION.md) | 从仅支持圆轨迹的 API 迁移 |
| [docs/video/README.md](docs/video/README.md) | 动图视频的渲染方式 |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 |

在 CUDA 主机上运行测试：`python -m pytest tests/ -q`。

## 引用

如需引用本软件和技术报告，请使用以下 BibTeX：

```bibtex
@software{diffct2025,
  author    = {Yipeng Sun},
  title     = {diffct: Differentiable Computed Tomography Reconstruction with CUDA},
  year      = 2025,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.14999333},
  url       = {https://doi.org/10.5281/zenodo.14999333}
}

@article{202605.1446,
  doi       = {10.20944/preprints202605.1446.v1},
  url       = {https://doi.org/10.20944/preprints202605.1446.v1},
  year      = 2026,
  month     = {May},
  publisher = {Preprints},
  author    = {Yipeng Sun and Linda-Sophie Schneider and Chengze ye and Andreas Maier},
  title     = {diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories},
  journal   = {Preprints}
}
```

## 许可与致谢

许可：[Apache 2.0](LICENSE)。本项目参考了 [PYRO-NN](https://github.com/csyben/PYRO-NN) 和 [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT)。欢迎提交 issue 和 pull request。核桃数据采用 CC BY 4.0 许可（Meaney 2022），见 [NOTICE](NOTICE)。
