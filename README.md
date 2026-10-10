<h1 align="center">diffct</h1>

<p align="center">English · <a href="https://github.com/sypsyp97/diffct/blob/main/README.zh.md">简体中文</a></p>

<p align="center">
  Differentiable CUDA projectors for CT: arbitrary trajectories, multi-GPU, multi-node, geometry gradients.
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.14999333"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square" alt="DOI"></a>
  <a href="https://pypi.org/project/diffct/"><img src="https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://sypsyp97.github.io/diffct/"><img src="https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=docs&style=flat-square" alt="Documentation"></a>
  <a href="https://github.com/sypsyp97/diffct/actions"><img src="https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/ci.yml?branch=main&label=CI&style=flat-square" alt="CI/CD"></a>
  <a href="https://deepwiki.com/sypsyp97/diffct"><img src="https://raw.githubusercontent.com/sypsyp97/diffct/main/docs/assets/deepwiki-badge.svg" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/sypsyp97/diffct/edf7c52227ec2cc14c988918faa7313f8ee0fe03/docs/assets/diffct_intro.gif" width="100%" alt="diffct intro: projection, sinogram, trajectories, multi-GPU split, reconstruction and geometry calibration">
</p>

<p align="center">
  <a href="https://github.com/sypsyp97/diffct/blob/edf7c52227ec2cc14c988918faa7313f8ee0fe03/docs/assets/diffct_intro.mp4">Intro video (MP4)</a> ·
  <a href="https://www.preprints.org/manuscript/202605.1446/v1">Technical report</a> ·
  <a href="https://doi.org/10.20944/preprints202605.1446.v1">DOI</a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#why-diffct">Features</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#citation">Citation</a>
</p>

> **Install** with the CUDA extra, `pip install "diffct[cu12]"` or `pip install "diffct[cu13]"`, after PyTorch. A plain `pip install diffct` needs a system CUDA Toolkit.
>
> **Version 2.0** replaces the circular-orbit API of 1.x with per-view trajectories. Code written for 1.x needs changes; see the [migration notes](https://github.com/sypsyp97/diffct/blob/main/docs/MIGRATION.md).
> For Apple Silicon, use the [MLX port by Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider/DiffCT-MLX).

## Why diffct

To our knowledge, diffct is the only open-source GPU CT library that combines arbitrary per-view trajectories, autograd through the volume, first-order gradients through the acquisition geometry, and built-in multi-GPU and multi-node execution. We compared it with LEAP, TIGRE, ASTRA/tomosipo, DiffDRR and others in October 2026.

- **Arbitrary trajectories.** Each view has its own source, detector centre and detector axes. Circular, helical, saddle, sinusoidal, random and calibrated scans use the same code. Trajectory tensors with `requires_grad=True` receive geometry gradients for calibration.
- **Matched operators.** `project()` and `backproject()` form an exact adjoint pair for the cell-constant Siddon model. Both support PyTorch autograd, with volume/sinogram gradients and Hessian-vector products.
- **Multi-GPU and multi-node.** Use `devices=[0, 1, 2, 3]` in one process, or one process per GPU with torchrun and NCCL. Views are partitioned and the volume is replicated. Speedup depends on workload and communication costs.
- **Analytical helpers.** `diffct.analytical` provides ramp filters (ram-lak, shepp-logan, cosine, hamming, hann), fan, cone and Parker weights, and FBP and FDK backprojection.

Capabilities, limits and isocenter rules: [docs/REFERENCE.md](https://github.com/sypsyp97/diffct/blob/main/docs/REFERENCE.md#capabilities-and-limits).

## Install

You need an NVIDIA GPU. Install PyTorch for your CUDA version first. Then install diffct with the extra for the same CUDA major version:

```bash
pip install "diffct[cu12]"    # PyTorch built for CUDA 12
pip install "diffct[cu13]"    # PyTorch built for CUDA 13
```

The extra installs the CUDA compiler libraries that Numba CUDA needs to compile the kernels. A plain `pip install diffct` does not install them. Use it only if a CUDA Toolkit is already installed on the system.

From source, with the examples:

```bash
git clone https://github.com/sypsyp97/diffct.git
cd diffct && pip install -e ".[cu12]"
python examples/quickstart.py               # smoke test: prints adjoint mismatch ~1e-8 for each beam
```

<details>
<summary>Notes on CUDA versions</summary>

- Numba CUDA imports `numpy.row_stack`, which NumPy 2.5 removes. Keep
  `numpy<2.5`.
- Keep NVVM and NVJitLink compatible with the CUDA libraries that PyTorch
  loads. A newer NVVM with an older NVJitLink fails when the kernels compile.
- A tested CUDA 13 set:
  `pip install "numpy<2.5" "numba-cuda[cu13]" "cuda-toolkit[cccl,cudart,nvrtc,nvvm]==13.0.2" "nvidia-nvjitlink<13.1"`.
- A tested CUDA 12 set: PyTorch 2.10 (cu126) with `numba-cuda[cu12]` 0.30.4 on
  NVIDIA driver 535.

</details>

## Quickstart

Create the acquisition once. Then call `project()` and `backproject()`.

```python
import torch
from diffct import Projector, spiral_trajectory_3d

trajectory = spiral_trajectory_3d(360, sid=320.0, sdd=512.0, z_range=40.0, n_turns=1.0, device="cpu")
A = Projector(trajectory, volume_shape=(128, 128, 128), detector_shape=(384, 256), detector_spacing=0.8)

volume = torch.rand(A.volume_shape, device="cuda")
sinogram = A.project(volume)        # (360, 384, 256) line integrals
adjoint = A.backproject(sinogram)   # matched adjoint A^T

x = torch.zeros_like(volume, requires_grad=True)
loss = 0.5 * (A.project(x) - sinogram).square().sum()
loss.backward()                     # x.grad = A^T (A x - y)
```

**Custom or calibrated trajectories.** Pass calibrated tensors directly instead of a generator; no circular fit is needed. See [REFERENCE](https://github.com/sypsyp97/diffct/blob/main/docs/REFERENCE.md#custom-or-calibrated-trajectories).

**Geometry gradients.** Set `requires_grad=True` on trajectory tensors before you build `Projector`. See [REFERENCE](https://github.com/sypsyp97/diffct/blob/main/docs/REFERENCE.md#geometry-gradients).

**Several GPUs and nodes.** Use `devices=[0, 1, 2, 3]` in one process, or one process per GPU with torchrun for one or more nodes. See [REFERENCE](https://github.com/sypsyp97/diffct/blob/main/docs/REFERENCE.md#several-gpus-and-nodes) and [DISTRIBUTED.md](https://github.com/sypsyp97/diffct/blob/main/docs/DISTRIBUTED.md).

```python
A = Projector(trajectory, (128, 128, 128), (384, 256), detector_spacing=0.8, devices=[0, 1, 2, 3])
```

## Performance

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sypsyp97/diffct/main/docs/assets/scaling_dark.png">
    <img src="https://raw.githubusercontent.com/sypsyp97/diffct/main/docs/assets/scaling_light.png" width="100%" alt="Time per forward, adjoint and CGLS iteration on 1, 4 and 8 GPUs, 64^3 and 128^3">
  </picture>
</p>

| 128³ volume | 1 GPU | 4 GPUs, 1 node | 8 GPUs, 2 nodes |
|---|---:|---:|---:|
| Forward projection | 8.45 ms | 2.66 ms (3.2×) | 1.68 ms (5.0×) |
| Adjoint (backprojection) | 20.66 ms | 5.89 ms (3.5×) | 4.62 ms (4.5×) |
| CGLS iteration | 30.08 ms | 8.08 ms (3.7×) | 5.59 ms (5.4×) |

Circular trajectory, 360 views, detector (2n, 1.5n) cells at pitch 1.25, A100 64 GB GPUs. Small volumes scale less: at 64³, one CGLS iteration takes 4.63 ms on 1 GPU, 1.57 ms on 4 and 1.56 ms on 8. Raw data: [docs/assets/scaling.json](https://github.com/sypsyp97/diffct/blob/main/docs/assets/scaling.json).

To measure one against several GPUs on your machine (default scan of the script, not the table setup):

```bash
python examples/benchmark_projector.py --devices 0 1
```

## Gallery

TV denotes TV-regularized iterative reconstruction; Adam is the optimizer. Figure labels distinguish the method from its optimizer.

**Measured walnut.** 240 measured views, circular cone beam, 256³: FDK (Hann window), SIRT (200 iterations), CGLS (20), TV (300, weight 0.3); axial and coronal centre slices.

<p align="center">
  <img src="https://raw.githubusercontent.com/sypsyp97/diffct/main/docs/assets/walnut_measured.png?v=e0092c4" width="100%" alt="Measured walnut: FDK, SIRT, CGLS and TV reconstructions">
</p>

```bash
python examples/walnut_reconstruction.py --figure out.png
```

**Simulated helical scan of the walnut.** The FDK walnut volume as ground truth, 720 views, 1% noise, 256³. PSNR: FDK 26.67 dB, CGLS (30 iterations) 34.13 dB, SIRT (200) 34.52 dB, TV (200, weight 1.0) 37.77 dB.

<p align="center">
  <img src="https://raw.githubusercontent.com/sypsyp97/diffct/main/docs/assets/walnut_helical.png?v=e0092c4" width="100%" alt="Simulated helical scan of the walnut: FDK, CGLS, SIRT and TV reconstructions">
</p>

```bash
python examples/walnut_reconstruction.py --algorithms --save-volume walnut256.npy
python examples/iterative_reconstruction.py --size 256 --views 720 --trajectory helical --noise 0.01 --phantom walnut256.npy --figure out.png
```

Walnut data: Meaney 2022, Zenodo 6986012, CC BY 4.0; see [examples/data/NOTICE](https://github.com/sypsyp97/diffct/blob/main/examples/data/NOTICE).

## Examples

`quickstart.py`, `analytical_reconstruction.py`, `iterative_reconstruction.py`, `walnut_reconstruction.py`, `geometry_calibration.py`, `benchmark_projector.py`, `plot_trajectory.py`, and the Slurm template `slurm/multi_node.sbatch`. Launch modes and distributed-loss rules are in [examples/README.md](https://github.com/sypsyp97/diffct/blob/main/examples/README.md).

## Documentation

| Document | Contents |
|---|---|
| [Trajectory guide](https://github.com/sypsyp97/diffct/blob/main/docs/source/trajectories.rst) | Trajectory tuples and `Projector` usage |
| [docs/REFERENCE.md](https://github.com/sypsyp97/diffct/blob/main/docs/REFERENCE.md) | Capabilities and limits, custom trajectories, geometry gradients, multi-GPU details, geometry and units |
| [docs/DISTRIBUTED.md](https://github.com/sypsyp97/diffct/blob/main/docs/DISTRIBUTED.md) | Execution/memory choices, distributed loss rules, Slurm and cross-node checks |
| [docs/MIGRATION.md](https://github.com/sypsyp97/diffct/blob/main/docs/MIGRATION.md) | Moving from the circular-only API |
| [docs/video/README.md](https://github.com/sypsyp97/diffct/blob/main/docs/video/README.md) | How the intro video is rendered |
| [CHANGELOG.md](https://github.com/sypsyp97/diffct/blob/main/CHANGELOG.md) | Release history |

Run the test suite on a CUDA host with `python -m pytest tests/ -q`.

## Citation

If you use this library in your research, please cite the software and the technical report:

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

## License and acknowledgements

Apache 2.0, see [LICENSE](https://github.com/sypsyp97/diffct/blob/main/LICENSE). The project draws on
[PYRO-NN](https://github.com/csyben/PYRO-NN) and
[geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT).
Issues and pull requests are welcome. The walnut data are CC BY 4.0 (Meaney 2022); see [NOTICE](https://github.com/sypsyp97/diffct/blob/main/NOTICE).
