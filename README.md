<h1 align="center">diffct</h1>

<p align="center">English · <a href="https://github.com/sypsyp97/diffct/blob/main/README.zh.md">简体中文</a></p>

<p align="center">
  <strong>Differentiable CT operators for measured scanner geometry.</strong><br>
  Project volumes, backproject measurements, reconstruct images and calibrate scanner geometry in PyTorch, from CPU experiments to multi-node CUDA execution.
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
  <img src="https://raw.githubusercontent.com/sypsyp97/diffct/main/docs/assets/diffct_intro.gif" width="100%" alt="diffct intro: projection, sinogram, trajectories, multi-GPU split, reconstruction and geometry calibration">
</p>

<p align="center">
  The video shows trajectories, autograd, geometry gradients, multi-GPU execution and reconstruction from measured walnut data · <a href="https://github.com/sypsyp97/diffct/blob/main/docs/assets/diffct_intro.mp4">MP4</a>.<br>
  Walnut data: Meaney 2022, Zenodo 6986012, CC BY 4.0; see <a href="https://github.com/sypsyp97/diffct/blob/main/examples/data/NOTICE">examples/data/NOTICE</a>.
</p>

<p align="center">
  <a href="#ct-operators">Operators</a> ·
  <a href="#reconstruction-methods">Reconstruction</a> ·
  <a href="#measured-helical-ct">Clinical CT</a> ·
  <a href="#multi-gpu-scaling">Scaling</a> ·
  <a href="#installation">Installation</a> ·
  <strong><a href="https://sypsyp97.github.io/diffct/">Documentation</a></strong>
</p>

> **Version 2.x** uses per-view trajectories. Code written for 1.x needs [migration](https://sypsyp97.github.io/diffct/migration.html). This README tracks the source checkout. Features under [Unreleased](https://github.com/sypsyp97/diffct/blob/main/CHANGELOG.md#unreleased) require installation from source.

## CT operators

diffct gives PyTorch a matched pair of CT projection operators. The trajectory tensors specify a ray direction or source position for each view, together with the detector centre and axes. Circular, helical, saddle, calibrated and measured clinical trajectories use the same tensor format.

To our knowledge, diffct is the only open-source GPU CT library with arbitrary per-view trajectories, autograd through the volume, first-order geometry gradients, and multi-node execution. We checked LEAP, TIGRE, ASTRA/tomosipo, DiffDRR and other libraries in October 2026.

- **Matched adjoint.** `project()` and `backproject()` implement the cell-constant Siddon model as an exact adjoint pair. `project()` supports gradients with respect to the input volume, and `backproject()` with respect to the input measurements. Both support Hessian-vector products.
- **Measured geometry.** Per-view source positions can include flying focal spot offsets. `ParameterizedSurface` describes cylindrical and other curved detectors without rebinning.
- **Reconstruction.** `cgls`, `sirt`, `tv_reconstruction`, differentiable FBP/FDK and gradient-based optimization share the same operators. Differentiable FBP/FDK can learn filters and weights. The projection operators can supply volume gradients for optimization.
- **Scale.** Automatic chunking, disk-backed state and distributed execution let diffct reconstruct scans that exceed one GPU's memory.

Technical report: [diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories](https://doi.org/10.20944/preprints202605.1446.v2).

## Reconstruction methods

The figure compares analytical and iterative reconstruction. The input is a noiseless 120-view fan-beam scan simulated from a clinical abdominal slice of Mayo liver case L213. The reference panel is the vendor image. The other panels show FBP and three iterative methods. SIRT uses 200 iterations, CGLS 30 iterations and TV-regularized reconstruction 300 iterations with weight 0.3.

<p align="center">
  <picture>
    <source srcset="docs/source/_static/gallery/readme_methods.png">
    <img src="https://raw.githubusercontent.com/sypsyp97/diffct/main/docs/source/_static/gallery/readme_methods.png" width="100%" alt="Reference patient slice and FBP, SIRT, CGLS and TV reconstructions">
  </picture>
</p>

<p align="center">
  Patient image source: Mayo Clinic LDCT-and-Projection-data, liver case L213, CC BY 4.0.
</p>

## Measured helical CT

diffct reconstructs the measured helical scan of liver case L213. The scan was acquired on a GE scanner and has 7,140 full-dose projection views on an 888 × 64 cylindrical detector. The rows show axial, coronal and sagittal sections. The columns compare the vendor image, SIRT, CGLS and TV-regularized reconstruction. SIRT uses 100 iterations, CGLS 15 iterations and TV 150 iterations. The two TV columns use weights 1 and 10. All reconstructions use 1 mm voxels.

<p align="center">
  <picture>
    <source srcset="docs/source/_static/gallery/readme_clinical.png">
    <img src="https://raw.githubusercontent.com/sypsyp97/diffct/main/docs/source/_static/gallery/readme_clinical.png" width="100%" alt="Measured helical CT case L213: vendor reference, SIRT, CGLS and TV">
  </picture>
</p>

Patient data: McCollough et al., *Low Dose CT Image and Projection Data (LDCT-and-Projection-data)*, Version 7, The Cancer Imaging Archive, 2020, [10.7937/9npb-2637](https://doi.org/10.7937/9npb-2637). Dataset paper: Moen et al., Medical Physics, 2020, [10.1002/mp.14594](https://doi.org/10.1002/mp.14594). Chest and Liver subsets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [docs/assets/gallery/NOTICE](https://github.com/sypsyp97/diffct/blob/main/docs/assets/gallery/NOTICE).

## Multi-GPU scaling

The benchmark uses a 512³ FP32 volume, A100 64 GB GPUs, a circular cone beam, 360 views, a 1024 × 768 detector and pitch 1.25.

| 512³ volume | 1 GPU | 4 GPUs, 1 node | 8 GPUs, 2 nodes |
|---|---:|---:|---:|
| Forward projection | 566.79 ms | 150.76 ms (3.8×) | 81.58 ms (6.9×) |
| Adjoint (backprojection) | 1060.77 ms | 279.37 ms (3.8×) | 224.04 ms (4.7×) |
| CGLS iteration | 1753.73 ms | 468.09 ms (3.7×) | 337.39 ms (5.2×) |

The [performance guide](https://sypsyp97.github.io/diffct/performance.html) gives the methodology, chart and raw samples. For more reconstructions, see the [examples](https://sypsyp97.github.io/diffct/examples.html).

## Installation

diffct needs Python 3.10 or newer. Install PyTorch for your device.

For native NVIDIA CUDA execution, check the CUDA version of the PyTorch build:

```bash
python -c "import torch; print(torch.version.cuda)"
```

When `torch.version.cuda` is 12.x, run:

```bash
pip install "diffct[cu12]"
```

When `torch.version.cuda` is 13.x, run:

```bash
pip install "diffct[cu13]"
```

For CPU, PyTorch ROCm or Apple Silicon, install diffct without an extra:

```bash
pip install diffct
```

Select `backend="torch"` for the portable backend. Apple Silicon uses the built-in MPS support. A community [MLX port by Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider/DiffCT-MLX) also exists. CUDA compatibility notes and source installation: [installation guide](https://sypsyp97.github.io/diffct/installation.html).

## Quickstart

Create one projector. Project the image and backproject the measurements. Then compute the loss gradient with respect to the image estimate:

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

Expected output: `torch.Size([180, 96]) torch.Size([64, 64])`. CPU and Apple Silicon variants are in the [quickstart guide](https://sypsyp97.github.io/diffct/quickstart.html).

## Documentation

[Quickstart](https://sypsyp97.github.io/diffct/quickstart.html) · [Trajectories and geometry](https://sypsyp97.github.io/diffct/trajectories.html) · [Clinical helical CT](https://sypsyp97.github.io/diffct/clinical_helical_example.html) · [Mayo public datasets](https://sypsyp97.github.io/diffct/mayo_datasets_example.html) · [Multi-GPU](https://sypsyp97.github.io/diffct/multi_gpu.html) · [API reference](https://sypsyp97.github.io/diffct/api.html) · [Examples](https://sypsyp97.github.io/diffct/examples.html)

## Citation

If you use diffct in your research, cite both the software and the technical report. Machine-readable metadata is in [CITATION.cff](https://github.com/sypsyp97/diffct/blob/main/CITATION.cff).

<details open>
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

## License and acknowledgements

Apache 2.0, see [LICENSE](https://github.com/sypsyp97/diffct/blob/main/LICENSE). The project draws on [PYRO-NN](https://github.com/csyben/PYRO-NN) and [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT). Issues and pull requests are welcome.
