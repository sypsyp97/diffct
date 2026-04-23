# diffct: Differentiable Computed Tomography Operators

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](https://sypsyp97.github.io/diffct/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=CI&style=flat-square)](https://github.com/sypsyp97/diffct/actions)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/diffct)

🌏 **Language**: English | [简体中文](README.zh.md)

A high-performance, CUDA-accelerated library for circular orbits CT
reconstruction with end-to-end differentiable operators, amplitude-
calibrated analytical FBP / FDK, and a separable-footprint projector
family for cell-integrated forward models. Built for optimization and
deep-learning integration.

⭐ **Please star this project if you find it is useful!**

**Maintenance:** DiffCT is now maintained by
[Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider) at
[Linda-SophieSchneider/DiffCT-MLX](https://github.com/Linda-SophieSchneider/DiffCT-MLX).

## 🔀 Branches

### Main Branch (Stable, PyPI)
This is the **stable release** branch supporting circular-orbit CT
reconstruction. Every versioned release on
[PyPI](https://pypi.org/project/diffct/) comes from `main`. See
[CHANGELOG.md](CHANGELOG.md) for the full release history.

### Dev Branch (arbitrary trajectories)
The `dev` branch is the arbitrary-trajectory evolution of the library.
Kernels take per-view ``(src_pos, det_center, det_u_vec[, det_v_vec])``
arrays instead of closed-form `sdd / sid / beta` scalars, so you can
reconstruct along **spiral, saddle, sinusoidal, or any user-supplied
trajectory**. It is kept in sync with `main`'s 1.2.11 analytical
reconstruction overhaul (ramp filter, weighted backproject, tests,
benchmark suite); the only thing currently deferred from `main` is the
1.3.0 separable-footprint (SF) backends, because generalising the
trapezoidal footprint math to arbitrary trajectories is a separate
research effort.

⚠️ **Note:** The dev branch is under active development and is not
published to PyPI. If you find any bugs please
[raise an issue](https://github.com/sypsyp97/diffct/issues).

## ✨ Features

- **Fast:** CUDA-accelerated forward and backward projectors with
  Numba CUDA kernels, plus dedicated voxel-driven FBP / FDK gather
  kernels with coalesced memory writes.
- **Differentiable:** End-to-end gradient propagation via
  ``torch.autograd``. Every projector / backprojector pair is a
  byte-accurate adjoint, verified by
  ``tests/test_adjoint_inner_product.py`` and
  ``torch.autograd.gradcheck`` in ``tests/test_gradcheck.py``.
- **Analytical reconstruction:** Amplitude-calibrated FBP / FDK
  pipelines via ``ramp_filter_1d`` (Ram-Lak / Hann / Hamming /
  cosine / Shepp-Logan windows, configurable padding and physical
  ``sample_spacing``), ``fan_cosine_weights`` /
  ``cone_cosine_weights``, ``parker_weights``,
  ``angular_integration_weights``, and
  ``parallel_weighted_backproject`` / ``fan_weighted_backproject`` /
  ``cone_weighted_backproject``. A unit-density phantom reconstructs
  back to amplitude 1 without any manual scaling.
- **Separable-footprint projectors:** Optional ``backend="sf"``
  (fan) and ``backend="sf_tr"`` / ``"sf_tt"`` (cone) selectors on
  every ``FanProjectorFunction`` / ``ConeProjectorFunction`` call
  expose voxel-driven SF projectors (Long-Fessler-Balter, IEEE TMI
  2010). The matched SF forward model is a cell-integrated
  alternative to ray-sampled Siddon and is useful when an iterative
  or learned pipeline wants a **mass-conserving** forward operator.
  The analytical FBP / FDK gather (``backend="sf"`` on
  ``fan_weighted_backproject`` / ``cone_weighted_backproject``) uses
  LEAP's chord-weighted matched-adjoint form
  (``projectors_SF.cu``); on standard Shepp-Logan phantoms it
  reaches Siddon-VD parity in both amplitude and MSE (within ~1 %).
  See the Core Algorithm section below for when SF is worth the
  extra ~2-3x forward cost.
- **Tested:** 71 pytest tests covering adjoint identity, gradcheck,
  smoke, FBP / FDK accuracy per geometry, detector / center offsets,
  and 27 ramp-filter window cases. Opt-in 27-case
  ``pytest-benchmark`` perf suite under ``tests/benchmarks/`` for
  before/after regression tracking.

## 📐 Supported Geometries

- **Parallel Beam:** 2D parallel-beam geometry
- **Fan Beam:** 2D fan-beam geometry
- **Cone Beam:** 3D cone-beam geometry

## 🔬 Core Algorithm

At the heart of every `diffct` projector / backprojector pair is
**Siddon's algorithm** ([Siddon 1985](https://doi.org/10.1118/1.595715))
— a ray-driven integer DDA that walks each ray through the voxel
grid, stepping forward only at voxel boundaries and giving exact
parametric intersection lengths in `O(N)` per ray.

`diffct`'s Siddon kernels implement **cell-constant segment
integration**: each ray integral is approximated as
`Σ Δt_m · f_{cell(m)}`, i.e. every traversed pixel (2D) or voxel
(3D) contributes its value weighted by the exact chord length
`Δt_m` that the ray spends inside that cell, with no sub-cell
interpolation. The voxel-driven FBP / FDK gather backprojector
(`*_weighted_backproject`) on the analytical side is a separate
path and does still bilinearly sample the filtered sinogram at each
voxel's projected detector footprint. The matched autograd adjoint
of the Siddon kernels scatters each ray-domain gradient back into
the same traversed cell with the same `Δt_m`, giving a byte-
accurate adjoint verified by `<Ax, y> ≈ <x, A^T y>` at float32
precision (see `tests/test_adjoint_inner_product.py`).

**Why this works for autograd.** The ray integral
`Σ Δt_m · f_{cell(m)}` is linear in the image voxel values `f`, so
`∂sinogram / ∂voxel` is simply the sum of the `Δt_m` weights
contributed by every ray-segment that traverses that voxel — well-
defined, non-zero, and trivially matched by the adjoint scatter.
`torch.autograd` flows gradients back through the projector without
surrogate tricks or straight-through estimators. A single integer-
DDA kernel also keeps the forward and adjoint code structurally
identical across parallel, fan, and cone, which is what makes the
adjoint byte-accurate in the first place.

**Sharpness and the ramp filter window.** Cell-constant Siddon is a
thin-ray point-like sampler in the image domain, so in a full
analytical reconstruction (forward -> ramp filter -> voxel-driven
gather) high-frequency content passes through the forward side
essentially unfiltered and lands in the ramp filter. The sharpness
/ ringing trade-off is therefore controlled at the ramp window
stage:

- **Ramp filter window**: `ramp_filter_1d(window=...)` picks the
  frequency-domain apodization applied on top of the ramp. In order
  of sharpness: `"ram-lak"` > `"hamming"` > `"hann"`. Sharper
  windows recover more high frequency content at the cost of more
  visible ringing / noise. This is by far the biggest knob on the
  reconstruction MTF at typical CBCT geometries.

**Separable-footprint (SF) backend — honest scope.** The optional
`backend="sf"` (fan) and `"sf_tr"` / `"sf_tt"` (cone) selectors on
`fan_weighted_backproject` and `cone_weighted_backproject` replace
the default bilinear voxel-driven gather with a gather that
integrates the filtered sinogram over each voxel's projected
trapezoidal footprint, in LEAP's chord-weighted matched-adjoint
form (`projectors_SF.cu`). What you actually get, verified on
Shepp-Logan and the walnut dataset:

- **Amplitude:** matches Siddon VD to within ~1 % at nominal,
  sub-nominal (`voxel = 0.5 * detector_pitch * sid / sdd`) and
  mildly supra-nominal voxel grids. Both backends are consistent
  with the same analytical FBP / FDK scale constant on unit-
  density phantoms.
- **MSE / SSIM:** SF is typically a whisker better than VD
  (fractions of a percent) on standard phantoms; do not expect a
  noticeable MSE win from the backend alone.
- **Visible MTF:** at the ~1.5-3x CBCT magnifications used in
  fan / cone examples, SF and VD produce **visually
  indistinguishable** edge profiles. Plot a line through a sharp
  edge and the two curves trace each other. The "SF is sharper at
  sub-nominal" claim you see in the SF literature is real in the
  extreme-sub-nominal regime (voxel much smaller than one detector
  bin) but invisible at the geometries in the shipped examples.

So why ship the SF backend at all? Because the **forward** side is
the actually interesting one:

- SF forward is **mass-conserving** per voxel — a single voxel's
  contribution is spread across the correct multi-bin footprint
  instead of concentrated at one bin, which matters for iterative
  reconstruction, learned priors, and any loss that compares
  sinograms directly. Cell-constant Siddon forward is a thin-ray
  point sampler and only conserves mass statistically.
- SF's matched adjoint is byte-accurate (verified by
  `tests/test_adjoint_inner_product.py`), so gradients flow
  correctly through the cell-integrated forward model.
- The SF-matched autograd adjoint and the LEAP chord-weighted FBP
  gather live in different kernels: the first is used by
  `FanProjectorFunction` / `ConeProjectorFunction` on the backward
  pass, the second is used by `fan_weighted_backproject` /
  `cone_weighted_backproject` when you pick the SF backend. Both
  are exposed; picking `backend="sf"` on both sides gives you a
  consistent cell-integrated pipeline.

**Bottom line.** If you just want the cleanest FBP / FDK of a
Shepp-Logan or a walnut on typical CBCT geometry, stay on
`backend="siddon"` and tune the ramp window. Switch to
`backend="sf"` / `"sf_tr"` / `"sf_tt"` when you care about the
*forward model* being cell-integrated — iterative reconstruction,
learned priors, sinogram-level losses, or comparing against a LEAP
baseline. Concrete usage of both paths is in `examples/fbp_fan.py`,
`examples/fdk_cone.py` and `examples/realdata_walnut_fdk.py`.

## 🥜 Real-data Example

Cone-beam FDK reconstruction of a real walnut from the Helsinki
industrial CBCT scanner ([Meaney 2022, Zenodo
10.5281/zenodo.6986012](https://doi.org/10.5281/zenodo.6986012),
CC-BY 4.0). The shipped sinogram at
[`examples/data/walnut_cone.npz`](examples/data/walnut_cone.npz)
is a 241-view, 256x256-per-view, flat-field-normalised and logged
subset of the original 721 x 2368 x 2240 uint16 acquisition (binned
8x, cropped 256x256, float16, ~25 MB). A sample reconstruction
montage is shipped at
[`examples/data/walnut_reco.png`](examples/data/walnut_reco.png).
Run the full analytical FDK pipeline with

```bash
python examples/realdata_walnut_fdk.py
```

The same analytical wrappers (`cone_cosine_weights`, `ramp_filter_1d`,
`angular_integration_weights`, `cone_weighted_backproject`) that
reconstruct Shepp-Logan phantoms in `fdk_cone.py` are used here with
no algorithmic changes — only the geometry is swapped for the one
stored in the `.npz`. The example reconstructs at half the nominal
voxel size on a 512³ grid with `backend="sf_tr"` and a Hamming
ramp window; you can switch to `backend="siddon"` and get a
visually equivalent result (the backend choice is a forward-model
preference here, not a reconstruction-sharpness knob). See
[`examples/data/NOTICE`](examples/data/NOTICE) for the full
attribution and the regeneration procedure.

## 🧩 Code Structure

```bash
diffct/
├── diffct/
│   ├── __init__.py            # public API re-exports
│   └── differentiable.py      # CUDA kernels, autograd Functions,
│                              # analytical helpers, SF backends
├── examples/                  # circular-orbit example scripts
│   ├── fbp_parallel.py
│   ├── fbp_fan.py             # with Parker short-scan switch
│   ├── fdk_cone.py            # with Parker short-scan switch
│   ├── iterative_reco_parallel.py
│   ├── iterative_reco_fan.py
│   ├── iterative_reco_cone.py
│   ├── realdata_fbp_parallel.py  # synthetic real-data pipeline
│   ├── realdata_fbp_fan.py       #   (Beer-Lambert + Poisson + -log)
│   ├── realdata_fdk_cone.py
│   ├── realdata_walnut_fdk.py    # real Helsinki walnut CBCT data
│   └── data/
│       ├── walnut_cone.npz    # ~25 MB preprocessed walnut sinogram
│       ├── walnut_reco.png    # sample FDK reconstruction montage
│       ├── preprocess_walnut.py  # regenerator from Zenodo source
│       └── NOTICE             # CC-BY 4.0 attribution
├── tests/
│   ├── test_*.py              # adjoint / gradcheck / accuracy /
│   │                          # offsets / weights / ramp-filter
│   └── benchmarks/            # opt-in pytest-benchmark perf suite
├── docs/                      # Sphinx documentation sources
├── pyproject.toml             # project metadata
├── pytest.ini
├── CHANGELOG.md               # Keep-a-Changelog release notes
├── README.md                  # English README (this file)
├── README.zh.md               # 简体中文 README
└── LICENSE
```

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU
- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/), [NumPy](https://numpy.org/), [Numba](https://numba.readthedocs.io/en/stable/user/installing.html), [CUDA](https://developer.nvidia.com/cuda-toolkit)

### Installation

**CUDA 12 (Recommended):**
```bash
# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 12.8.1 as example) PyTorch, and Numba
conda install nvidia/label/cuda-12.8.1::cuda-toolkit

# Install Pytorch, you can find the commend here: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 12
pip install numba-cuda[cu12]

# Install diffct
pip install diffct
```

<details>
<summary>CUDA 13 Installation</summary>

```bash
# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 13.0.2 as example) PyTorch, and Numba
conda install nvidia/label/cuda-13.0.2::cuda-toolkit

# Install Pytorch, you can find the commend here: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 13
pip install numba-cuda[cu13]

# Install diffct
pip install diffct
```

</details>

<details>
<summary>CUDA 11 Installation</summary>

```bash
# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 11.8.0 as example) PyTorch, and Numba
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

# Install Pytorch, you can find the commend here: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 11
pip install numba-cuda[cu11]

# Install diffct
pip install diffct
```

</details>

### Running the tests

```bash
pytest tests/ -q                             # 71 tests, ~15 s
pytest tests/benchmarks/ --benchmark-only    # opt-in perf suite, requires pytest-benchmark
```

## 📝 Citation

If you use this library in your research, please cite:

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

## 📄 License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This project was highly inspired by:

- [PYRO-NN](https://github.com/csyben/PYRO-NN)
- [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT)
- [LEAP](https://github.com/LLNL/LEAP) (LLNL / Hyojin Kim et al.)
  — the LEAP chord-weighted matched-adjoint form in
  [`projectors_SF.cu`](https://github.com/LLNL/LEAP/blob/main/src/projectors_SF.cu)
  is the reference implementation that the 1.3.1 analytical SF
  backprojection kernels (`_fan_2d_sf_fbp_backproject_kernel`,
  `_cone_3d_sf_tr_fdk_backproject_kernel`,
  `_cone_3d_sf_tt_fdk_backproject_kernel`) in diffct are ported
  from. Apache 2.0-licensed, big thanks to the LEAP team.

Issues and contributions are welcome!
