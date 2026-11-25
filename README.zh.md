<h1 align="center">diffct</h1>

<p align="center"><a href="https://github.com/sypsyp97/diffct/blob/main/README.md">English</a> · 简体中文</p>

<p align="center">
  <strong>把真实 CT 扫描几何放进 PyTorch 训练循环。</strong><br>
  diffct 提供可微 CT 投影和匹配反投影算子，支持任意扫描轨迹——从 CPU 实验到多节点 CUDA 重建。
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.14999333"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square" alt="DOI"></a>
  <a href="https://pypi.org/project/diffct/"><img src="https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://sypsyp97.github.io/diffct/"><img src="https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=docs&style=flat-square" alt="Documentation"></a>
  <a href="https://github.com/sypsyp97/diffct/actions"><img src="https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/ci.yml?branch=main&label=CI&style=flat-square" alt="CI/CD"></a>
</p>

<p align="center">
  <img src="docs/source/_static/gallery/gradient_walnut.png?v=497749b" width="100%" alt="实测核桃的梯度重建结果随优化步数逐渐清晰">
</p>

<p align="center">
  通过投影算子优化得到的实测核桃梯度重建结果 · <a href="https://github.com/sypsyp97/diffct/blob/main/docs/assets/diffct_intro.mp4">概览视频</a>。<br>
  核桃数据：Meaney 2022，Zenodo 6986012，CC BY 4.0，见 <a href="examples/data/NOTICE">examples/data/NOTICE</a>。
</p>

<p align="center">
  <a href="#为什么使用-diffct">为什么使用 diffct</a> ·
  <a href="#安装">安装</a> ·
  <a href="#快速上手">快速上手</a> ·
  <a href="#结果">结果</a> ·
  <a href="#下一步">下一步</a> ·
  <strong><a href="https://sypsyp97.github.io/diffct/">完整文档</a></strong>
</p>

> **2.x 版**使用逐视角轨迹；为 1.x 写的代码需要[迁移](https://sypsyp97.github.io/diffct/migration.html)（英文）。本 README 对应源码版本；[Unreleased](https://github.com/sypsyp97/diffct/blob/main/CHANGELOG.md#unreleased) 中列出的功能需要从源码安装。

## 为什么使用 diffct

diffct 面向在 PyTorch 中构建学习型重建、几何标定和反问题方法的研究者。它的设计核心是同一套能力组合：任意逐视角采集几何、对体数据和几何的自动求导，以及能扩展到实测临床扫描的执行能力。

- **PyTorch 原生算子。** `project()` 与 `backproject()` 对分片常数 Siddon 模型构成匹配的伴随对；两者都支持自动求导、Hessian 向量积和常规 PyTorch 训练循环。
- **真实扫描几何。** 圆轨迹、螺旋、鞍形、标定轨迹和实测临床轨迹使用同一套逐视角张量；柱面探测器和 flying focal spot 偏移无需重合并到圆轨迹。
- **重建与学习。** `cgls`、`sirt`、`tv_reconstruction`、可微 FBP/FDK 和基于梯度的优化共享同一套算子。
- **大体积扫描。** 自动分块、磁盘状态、多卡和多节点执行可超过单张 GPU 的显存限制。

技术报告：[diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories](https://doi.org/10.20944/preprints202605.1446.v2)。

## 安装

diffct 需要 Python 3.10 或更高版本，以及与设备匹配的 PyTorch。使用原生 NVIDIA CUDA 路径时，先安装 PyTorch，检查其 CUDA 版本，再安装对应 extra：

```bash
python -c "import torch; print(torch.version.cuda)"   # 12.x or 13.x

# Pick the extra matching torch.version.cuda:
pip install "diffct[cu12]"    # CUDA 12.x
pip install "diffct[cu13]"    # CUDA 13.x
```

CPU、PyTorch ROCm 或 Apple Silicon（MPS）使用普通 `pip install diffct`，并选择 `backend="torch"`。Apple Silicon 通过内置 MPS 支持运行；Linda-Sophie Schneider 还有一个社区 [MLX 移植版](https://github.com/Linda-SophieSchneider/DiffCT-MLX)。CUDA 兼容性说明和源码安装见[安装指南](https://sypsyp97.github.io/diffct/installation.html)（英文）。

## 快速上手

投影一幅图像，应用匹配伴随，并计算梯度：

```python
import torch
from diffct import Projector, circular_trajectory_2d_parallel

trajectory = circular_trajectory_2d_parallel(180, device="cpu")
A = Projector(trajectory, (64, 64), 96, beam="parallel",
              detector_spacing=1.0, voxel_spacing=1.0)

image = torch.zeros(A.volume_shape, device="cuda", dtype=torch.float32)
image[20:44, 20:44] = 1.0
measurements = A.project(image)        # (180, 96)
adjoint = A.backproject(measurements)  # (64, 64), not an inverse

estimate = torch.zeros_like(image, requires_grad=True)
loss = 0.5 * (A.project(estimate) - measurements).square().sum()
loss.backward()
print(measurements.shape, estimate.grad.shape)
```

预期输出：`torch.Size([180, 96]) torch.Size([64, 64])`。CPU 和 Apple Silicon 版本见[快速上手指南](https://sypsyp97.github.io/diffct/quickstart.html)（英文）。

## 结果

**实测临床螺旋 CT。** Mayo Clinic 2020 低剂量 CT 数据集（LDCT-and-Projection-data）的肝脏病例 L213（GE 扫描仪）：全剂量投影，7140 个视角，888 × 64 柱面探测器；diffct 以 1 mm 体素、无正则化 CGLS 重建。轴向、冠状面和矢状面切片显示第 30 次迭代，残差曲线持续到第 60 次。示例文档见 [临床螺旋 CT](https://sypsyp97.github.io/diffct/clinical_helical_example.html) 与 [Mayo 公开数据集](https://sypsyp97.github.io/diffct/mayo_datasets_example.html)（英文）。

<p align="center">
  <img src="docs/source/_static/gallery/helical_L213.png?v=497749b" width="100%" alt="实测临床螺旋 CT：肝脏病例 L213 的 CGLS 轴向、冠状面、矢状面切片和残差曲线">
</p>

患者数据：McCollough et al.，*Low Dose CT Image and Projection Data (LDCT-and-Projection-data)*，Version 7，The Cancer Imaging Archive，2020，[10.7937/9npb-2637](https://doi.org/10.7937/9npb-2637)；数据集论文：Moen et al.，Medical Physics，2020，[10.1002/mp.14594](https://doi.org/10.1002/mp.14594)。Chest 和 Liver 子集采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可，见 [docs/assets/gallery/NOTICE](docs/assets/gallery/NOTICE)。

**多卡扩展。** diffct 2.4.3，512³ 体数据，FP32，A100 64 GB GPU；圆形锥束，360 个视角，1024×768 探测器，pitch 1.25：

| 512³ 体数据 | 1 卡 | 4 卡，1 节点 | 8 卡，2 节点 |
|---|---:|---:|---:|
| 正投影 | 566.79 ms | 150.76 ms (3.8×) | 81.58 ms (6.9×) |
| 伴随（反投影） | 1060.77 ms | 279.37 ms (3.8×) | 224.04 ms (4.7×) |
| CGLS 迭代 | 1753.73 ms | 468.09 ms (3.7×) | 337.39 ms (5.2×) |

方法、图表和原始样本见[性能指南](https://sypsyp97.github.io/diffct/performance.html)（英文）；更多重建示例见[示例](https://sypsyp97.github.io/diffct/examples.html)（英文）。

## 下一步

| 我想…… | 入口 |
|---|---|
| 运行第一个投影和梯度 | [Quickstart](https://sypsyp97.github.io/diffct/quickstart.html)（英文） |
| 使用自定义、标定或螺旋轨迹 | [Trajectories and geometry](https://sypsyp97.github.io/diffct/trajectories.html)（英文） |
| 读取 Mayo 临床投影和图像 | [Mayo public datasets](https://sypsyp97.github.io/diffct/mayo_datasets_example.html)（英文） |
| 重建实测临床螺旋 CT | [Clinical helical CT](https://sypsyp97.github.io/diffct/clinical_helical_example.html)（英文） |
| 扩展到多卡、多节点或有限内存 | [Multi-GPU](https://sypsyp97.github.io/diffct/multi_gpu.html) · [Chunking](https://sypsyp97.github.io/diffct/chunking.html)（英文） |
| 用 autograd 训练滤波器、权重或体数据 | [Differentiable FBP/FDK](https://sypsyp97.github.io/diffct/analytical_learning.html) · [Gradient reconstruction](https://sypsyp97.github.io/diffct/gradient_reconstruction.html)（英文） |
| 查询执行规则或 API 细节 | [Execution](https://sypsyp97.github.io/diffct/execution.html) · [API reference](https://sypsyp97.github.io/diffct/api.html)（英文） |
| 浏览全部重建示例 | [Examples](https://sypsyp97.github.io/diffct/examples.html)（英文） |

## 引用

如需引用 diffct，请同时引用软件和技术报告。机器可读元数据见 [CITATION.cff](https://github.com/sypsyp97/diffct/blob/main/CITATION.cff)。

<details>
<summary>BibTeX</summary>

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
  doi       = {10.20944/preprints202605.1446.v2},
  url       = {https://doi.org/10.20944/preprints202605.1446.v2},
  year      = 2026,
  month     = {Oct},
  publisher = {Preprints},
  author    = {Yipeng Sun and Linda-Sophie Schneider and Chengze Ye and Andreas Maier},
  title     = {diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories},
  journal   = {Preprints}
}
```

</details>

## 许可与致谢

Apache 2.0，见 [LICENSE](https://github.com/sypsyp97/diffct/blob/main/LICENSE)。项目借鉴了 [PYRO-NN](https://github.com/csyben/PYRO-NN) 和 [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT)。欢迎提交 issue 和 pull request。
