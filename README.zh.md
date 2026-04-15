# diffct: 可微分计算机断层重建算子

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](https://sypsyp97.github.io/diffct/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=CI&style=flat-square)](https://github.com/sypsyp97/diffct/actions)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/diffct)

🌏 **Language**: [English](README.md) | 简体中文

一个面向圆轨道 CT 重建的高性能 CUDA 加速库。提供端到端
可微分的 projector / backprojector、幅度已标定的解析 FBP /
FDK pipeline,以及基于单元积分模型的可分离 footprint
(separable-footprint) projector 族。为优化问题和深度学习集成
而设计。

⭐ **如果你觉得这个项目有用,请点个 star!**

## 🔀 分支说明

### Main 分支(稳定版,发布到 PyPI)
这是 **稳定发布** 分支,支持圆轨道 CT 重建。所有发布到
[PyPI](https://pypi.org/project/diffct/) 的版本都来自 `main`。
完整发布历史见 [CHANGELOG.md](CHANGELOG.md)。

### Dev 分支(任意轨迹)
`dev` 分支是这个库面向任意轨迹的演进版本。kernel 接收逐视角的
`(src_pos, det_center, det_u_vec[, det_v_vec])` 数组,而不是
闭式的 `sdd / sid / beta` 标量,所以你可以在 **螺旋 (spiral)、
鞍形 (saddle)、正弦 (sinusoidal) 或任意用户自定义轨迹** 上做
重建。目前 `dev` 跟 `main` 的 1.2.11 解析重建重构(ramp filter、
weighted backproject、测试、benchmark 套件)保持同步;唯一暂缓
从 `main` 迁移过来的是 1.3.0 的 separable-footprint (SF) 后端 ——
把梯形 footprint 推广到任意轨迹是一个独立的 research effort。

⚠️ **注意**: dev 分支处于活跃开发状态,不会发布到 PyPI。发现
bug 请
[提 issue](https://github.com/sypsyp97/diffct/issues)。

## ✨ 功能特性

- **快**: 用 Numba CUDA 写的 forward / backward projector,外加
  专用的 voxel-driven FBP / FDK gather kernel,内存写合并
  (coalesced writes)。
- **可微分**: 通过 `torch.autograd` 实现端到端梯度反传。每对
  projector / backprojector 都是 byte-accurate 的 adjoint,在
  `tests/test_adjoint_inner_product.py` 和
  `tests/test_gradcheck.py` 里用 `torch.autograd.gradcheck` 验证。
- **解析重建**: 幅度已标定的 FBP / FDK pipeline,由
  `ramp_filter_1d`(支持 Ram-Lak / Hann / Hamming / cosine /
  Shepp-Logan window,可配置 padding 和物理 `sample_spacing`)、
  `fan_cosine_weights` / `cone_cosine_weights`、`parker_weights`、
  `angular_integration_weights`、`parallel_weighted_backproject` /
  `fan_weighted_backproject` / `cone_weighted_backproject` 组成。
  单位密度 phantom 重建回来就是 amplitude 1,不需要手动缩放。
- **Separable-footprint projectors**: 在
  `FanProjectorFunction` / `ConeProjectorFunction` 调用里可以
  通过 `backend="sf"`(fan) 或 `backend="sf_tr"` / `"sf_tt"`(cone)
  切换到 voxel-driven 的 SF projector (Long-Fessler-Balter, IEEE
  TMI 2010)。这是一个**质量守恒** (mass-conserving) 的单元积分
  forward model,在 iterative reco 和 learned pipeline 里比
  Siddon 的 ray-sampled 版本更贴近 sinogram 的物理含义。解析
  FBP / FDK 那一侧的 `backend="sf"` 走的是 LEAP 的 chord-weighted
  matched-adjoint 形式 (`projectors_SF.cu`),在 Shepp-Logan 上
  amplitude 和 MSE 都跟 Siddon VD 持平 (差异 <1 %)。何时真的值得
  用 SF 见下面的 Core Algorithm。forward 代价大约是 2-3 倍。
- **测试齐全**: 66 个 pytest 用例覆盖 adjoint identity、gradcheck、
  smoke、每种几何的 FBP / FDK 精度、detector / center offsets、
  以及 29 个 ramp filter window case。`tests/benchmarks/` 下有
  可选的 27 个 `pytest-benchmark` 性能用例用于前后对比。

## 📐 支持的几何

- **Parallel Beam**: 2D 平行束几何
- **Fan Beam**: 2D 扇形束几何
- **Cone Beam**: 3D 锥形束几何

## 🔬 核心算法

`diffct` 每一对 projector / backprojector 的核心都是
**Siddon 算法** ([Siddon 1985](https://doi.org/10.1118/1.595715))
—— 一个 ray-driven 的 integer DDA,让每条射线只在 voxel 边界处
步进,对每条射线在 `O(N)` 步内给出精确的参数化 intersection
长度,不会浪费时间走空 voxel。

经典 Siddon 在每个步进点取 *单元值* (nearest-neighbor)。
`diffct` 把这一步改成了 **bilinear (2D) / trilinear (3D) 插值** ——
在每个采样点对周围 voxel 顶点做插值。同样的插值在整条解析重建
链路上出现了两次: 一次是 Siddon forward projector 在图像上游走
采样,一次是 voxel-driven 的 FBP / FDK gather backprojector
(`*_weighted_backproject`) 在滤波后 sinogram 上按每个 voxel 投
影到探测器的 footprint 采样。对应的 autograd adjoint 用完全相同
的权重做 scatter,保证 `<Ax, y> ≈ <x, A^T y>` 在 float32 精度下
byte-accurate(见 `tests/test_adjoint_inner_product.py`)。

**为什么这样设计**。插值权重对 voxel 值是连续的,所以
`∂sinogram / ∂voxel` 处处良定义,`torch.autograd` 可以直接
把梯度穿透 projector 回流,不需要任何 surrogate 或 straight-
through estimator 的技巧。纯 nearest-neighbor Siddon 给的是
piecewise-constant 输出,单元内部梯度恒为零 —— 对一次性的
FBP / FDK 没关系,对 iterative reconstruction 和 learned
reconstruction 就彻底不可用了。统一的 integer-DDA kernel 还让
forward 和 adjoint 的代码结构在 parallel / fan / cone 三种
几何里保持同构,这也是 adjoint 能做到 byte-accurate 的前提。

**代价: 图像会稍微变糊**。bi / trilinear 插值本质上是在 voxel
网格之上再叠一个温和的低通滤波器,重建的有效 MTF 比单元积分
projector 滚降得早一点。想把这部分分辨率找回来,最主要的旋钮
是 ramp filter window:

- **Ramp filter window**: `ramp_filter_1d(window=...)` 选择在 ramp
  上叠加的频域 apodization。锐度排序:`"ram-lak"` > `"hamming"` >
  `"hann"`。越锐的 window 保留越多高频内容,代价是 ringing /
  噪声更明显。在常规 CBCT 几何下,这个是影响重建 MTF 最大的旋钮,
  远大于 projector 后端的选择。

**关于 Separable-footprint (SF) 后端 —— 说点实话**。
`fan_weighted_backproject` 和 `cone_weighted_backproject` 上的
`backend="sf"`(fan)/ `"sf_tr"` / `"sf_tt"`(cone) 把默认的
bilinear voxel-driven gather 换成"按 voxel 投到探测器的梯形
footprint 做积分"的 gather,走 LEAP 的 chord-weighted matched-
adjoint 形式(`projectors_SF.cu`)。我们在 Shepp-Logan 和真核桃
数据上实测的结果:

- **幅度**:在 nominal / sub-nominal (`voxel = 0.5 * detector_pitch
  * sid / sdd`) / 略 supra-nominal 三档 voxel size 下,SF 跟
  Siddon VD 的 amplitude 都能对上(差异 < 1 %)。两个后端在
  unit-density phantom 上都是幅度校准的,共用同一组解析 FBP / FDK
  scale 常数。
- **MSE / SSIM**:SF 比 VD 稍好一点点(零点几个百分点量级)。
  不要期待后端选择本身能带来明显的 MSE 提升。
- **肉眼可见的 MTF**:在 fan / cone 例子里常见的 1.5-3 倍放大率
  下,SF 和 VD 产生 **肉眼几乎无差别** 的 edge profile。在一条
  硬边缘上画 line profile,两条曲线完全重合。SF 文献里说的
  "sub-nominal 下 SF 更锐"在极端 sub-nominal(voxel 远小于单个
  探测器 bin)下是真的,但在我们 shipped example 的几何下看不到。

那 SF 后端为什么还要保留?因为真正有价值的是 **forward** 这一侧:

- SF forward 是 **mass-conserving** 的 —— 一个 voxel 的贡献会按
  正确的 multi-bin footprint 摊到整条梯形上,而不是像 Siddon
  bilinear 那样集中在一个 bin 的 point sample。iterative 重建、
  learned prior、任何在 sinogram 上直接算 loss 的 pipeline 都
  更愿意用这种 forward。
- SF 的 matched adjoint 是 byte-accurate 的(见
  `tests/test_adjoint_inner_product.py`),所以梯度可以正确地
  流过这个单元积分 forward model。
- SF-matched autograd adjoint 跟 LEAP chord-weighted FBP gather
  是两套不同的 kernel:前者在 `FanProjectorFunction` /
  `ConeProjectorFunction` 的 backward pass 上被调用,后者在
  `fan_weighted_backproject` / `cone_weighted_backproject` 选 SF
  后端时被调用。两个都暴露出来,在 forward 和 backproject 两侧
  都选 `backend="sf"` 就能得到一条完全的单元积分 pipeline。

**一句话结论**。如果你只是想在常规 CBCT 几何上 FBP / FDK 一个
Shepp-Logan 或者核桃,留在 `backend="siddon"` 调 ramp window 就好。
当你关心的是 **forward model 是单元积分** —— iterative 重建、
learned prior、sinogram loss、跟 LEAP 对齐的实验 —— 才切到
`backend="sf"` / `"sf_tr"` / `"sf_tt"`。两条路径的具体用法在
`examples/fbp_fan.py`、`examples/fdk_cone.py` 和
`examples/realdata_walnut_fdk.py` 里都有演示。

## 🥜 真实数据示例

用真实核桃 CBCT 数据做的锥形束 FDK 重建,扫描来自赫尔辛基
大学工业 CT 实验室([Meaney 2022, Zenodo
10.5281/zenodo.6986012](https://doi.org/10.5281/zenodo.6986012),
CC-BY 4.0)。仓库里预处理好的 sinogram 放在
[`examples/data/walnut_cone.npz`](examples/data/walnut_cone.npz),
是原始 721 x 2368 x 2240 uint16 采集的一个 241 视角、每视角
256x256、做过 flat-field 归一化并取过 `-log` 的子集(8x 分辨率
binning,中心 crop 256x256,float16 存储,约 25 MB)。一张
示例重建 montage 在
[`examples/data/walnut_reco.png`](examples/data/walnut_reco.png)。
运行完整的解析 FDK pipeline:

```bash
python examples/realdata_walnut_fdk.py
```

这个例子用的和 `fdk_cone.py` 里重建 Shepp-Logan 完全是同一套
解析 wrapper(`cone_cosine_weights`、`ramp_filter_1d`、
`angular_integration_weights`、`cone_weighted_backproject`),
没有任何算法层面的改动 —— 只是把几何参数换成 `.npz` 里存的
那一套。默认跑在 half-nominal 体素 + 512³ 网格 + `backend="sf_tr"`
+ Hamming window;如果你切到 `backend="siddon"`,会得到肉眼
几乎看不出差别的结果(后端选择在这里是一个 forward-model 的
偏好,不是 sharpness 旋钮)。完整的 attribution 和再生步骤见
[`examples/data/NOTICE`](examples/data/NOTICE)。

## 🧩 代码结构

```bash
diffct/
├── diffct/
│   ├── __init__.py            # 公共 API 重新导出
│   └── differentiable.py      # CUDA kernels、autograd Functions、
│                              # 解析 helpers、SF 后端
├── examples/                  # 圆轨道示例脚本
│   ├── fbp_parallel.py
│   ├── fbp_fan.py             # 带 Parker short-scan 开关
│   ├── fdk_cone.py            # 带 Parker short-scan 开关
│   ├── iterative_reco_parallel.py
│   ├── iterative_reco_fan.py
│   ├── iterative_reco_cone.py
│   ├── realdata_fbp_parallel.py  # 合成真实数据流水线
│   ├── realdata_fbp_fan.py       #   (Beer-Lambert + Poisson + -log)
│   ├── realdata_fdk_cone.py
│   ├── realdata_walnut_fdk.py    # 真核桃 CBCT 数据
│   └── data/
│       ├── walnut_cone.npz    # ~25 MB 预处理好的核桃 sinogram
│       ├── walnut_reco.png    # 示例 FDK 重建 montage
│       ├── preprocess_walnut.py  # 从 Zenodo 原始数据再生
│       └── NOTICE             # CC-BY 4.0 attribution
├── tests/
│   ├── test_*.py              # adjoint / gradcheck / 精度 /
│   │                          # offsets / weights / ramp-filter
│   └── benchmarks/            # 可选的 pytest-benchmark 性能套件
├── docs/                      # Sphinx 文档源
├── pyproject.toml             # 项目元数据
├── pytest.ini
├── CHANGELOG.md               # Keep-a-Changelog 风格 release notes
├── README.md                  # 英文 README
├── README.zh.md               # 中文 README (本文件)
└── LICENSE
```

## 🚀 快速开始

### 依赖

- 支持 CUDA 的 GPU
- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/)、[NumPy](https://numpy.org/)、[Numba](https://numba.readthedocs.io/en/stable/user/installing.html)、[CUDA](https://developer.nvidia.com/cuda-toolkit)

### 安装

**CUDA 12(推荐)**:
```bash
# 创建并激活 conda 环境
conda create -n diffct python=3.12
conda activate diffct

# 安装 CUDA toolkit (这里以 12.8.1 为例)
conda install nvidia/label/cuda-12.8.1::cuda-toolkit

# 安装 PyTorch,从 https://pytorch.org/get-started/locally/ 查对应命令

# 安装 CUDA 12 版的 Numba
pip install numba-cuda[cu12]

# 安装 diffct
pip install diffct
```

<details>
<summary>CUDA 13 安装</summary>

```bash
conda create -n diffct python=3.12
conda activate diffct

conda install nvidia/label/cuda-13.0.2::cuda-toolkit

# 安装 PyTorch,从 https://pytorch.org/get-started/locally/ 查对应命令

pip install numba-cuda[cu13]

pip install diffct
```

</details>

<details>
<summary>CUDA 11 安装</summary>

```bash
conda create -n diffct python=3.12
conda activate diffct

conda install nvidia/label/cuda-11.8.0::cuda-toolkit

# 安装 PyTorch,从 https://pytorch.org/get-started/locally/ 查对应命令

pip install numba-cuda[cu11]

pip install diffct
```

</details>

### 跑测试

```bash
pytest tests/ -q                             # 66 个测试,~15 s
pytest tests/benchmarks/ --benchmark-only    # 可选的性能套件,需要 pytest-benchmark
```

## 📝 引用

如果你在研究中使用了本库,请引用:

```bibtex
@software{diffct2025,
  author       = {Yipeng Sun},
  title        = {diffct: Differentiable Computed Tomography
                 Reconstruction with CUDA},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14999333},
  url          = {https://doi.org/10.5281/zenodo.14999333}
}
```

## 📄 许可证

本项目基于 Apache 2.0 许可证发布 —— 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目高度受以下项目启发:

- [PYRO-NN](https://github.com/csyben/PYRO-NN)
- [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT)
- [LEAP](https://github.com/LLNL/LEAP) (LLNL / Hyojin Kim 等)
  —— 1.3.1 起 diffct 解析 FBP / FDK 路径上的三个 SF 反投 kernel
  (`_fan_2d_sf_fbp_backproject_kernel`、
  `_cone_3d_sf_tr_fdk_backproject_kernel`、
  `_cone_3d_sf_tt_fdk_backproject_kernel`) 是按 LEAP
  [`projectors_SF.cu`](https://github.com/LLNL/LEAP/blob/main/src/projectors_SF.cu)
  里的 chord-weighted matched-adjoint 形式移植的。Apache 2.0 协议,
  感谢 LEAP 团队的参考实现。

欢迎提 issue 和 contribution!
