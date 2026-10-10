# diffct 2.0 introduction

The film preserves the upstream feature story while using a restrained light
palette, readable sans-serif type and green emphasis. The measured walnut is
present throughout: a rotating 3D CT surface opens/closes the film, the scan rig
orbits that surface, and the final comparison shows eight reconstruction slices.

## Render

From the repository root, with Manim, PyAV, Pillow and LaTeX available:

```powershell
python -m manim -ql --media_dir .validation/video-review/preview docs/video/diffct_intro.py DiffCTIntro
python docs/video/render.py
```

The final command renders 1920 × 1080 at 30 fps, normalizes segment timestamps to
constant frame rate, and exports `docs/assets/diffct_intro.mp4` and the matching
800 × 450, 8 fps README GIF. It also saves `timeline.json` beside the intermediate
render under `.validation/video-review/final/` for shot-by-shot review.
Manim CE 0.20.1 and PyAV 17.0.0 were used. Saved assets render without CUDA.
Layout bounds are checked during rendering.

## Story and coverage

1. **Opening:** rotating 3D walnut, project name, PyTorch and feature overview.
2. **Trajectories:** circular, helical, saddle, sinusoidal and discrete per-view
   poses; a source, detector and cone rays orbit a 3D specimen/volume boundary.
   The same `Projector` interface takes each trajectory.
3. **Autograd:** walnut slice and sinogram, projection and matched adjoint,
   inner-product identity, loss/backward code, volume/sinogram second derivatives.
4. **Geometry gradients:** a clearly labelled pose-correction schematic alongside
   the recorded calibration loss, angle error and first-order geometry scope.
5. **Multi-GPU:** partition views, replicate the volume, distribute to eight GPUs
   on two nodes, show local/distributed API usage and NCCL, then measured timings
   and speedups with workload and hardware stated.
6. **Measured reconstruction:** four real projections, then FDK / SIRT / CGLS /
   TV as a four-column, two-row axial/coronal comparison with iteration counts.
7. **Ending:** rotating walnut, installation (PyTorch prerequisite, CUDA extras),
   documentation and repository links.

## Data and provenance

The API and archived measurements were checked against upstream main
`50a897f8860346ef906d4c909db081452c411657` (2.0.2). This worktree is based on that upstream commit; library behavior is unchanged.

- `walnut_measured.npz` and `data2d.npz` are copied byte-for-byte from that commit.
  The former contains four measured cone-beam projections plus FDK, SIRT (200),
  CGLS (20), and TV (300, weight 0.3) axial/coronal slices at 256³. All reconstruction
  panels use [0, 1]. `data2d.npz` contains a walnut slice and a SciPy rotate/sum
  **display simulation** of a parallel sinogram; the film labels it accordingly.
- `calib_history.json` is the saved, separate 64³ calibration run. Its seven
  samples are plotted on a labelled log10 axis, with steps 0–149 (150 steps).
  Pose dots are a schematic, not recovered measurements or saved optimizer poses.
  Displayed angle errors come directly from this JSON. Geometry gradients are
  first order; volume/sinogram gradients also support second derivatives.
- `docs/assets/scaling.json` is byte-identical to upstream. The film reads its
  CGLS 128³/360-view timings and calculates speedup from them. Hardware:
  A100-SXM-64GB. The bars are archived benchmarks, not this workstation's timings.
- `walnut_surface.npz` is newly derived from the **real measured dataset**, not
  generated imagery. Upstream 2.0.2's `walnut_reconstruction.py` reconstructed all
  240 views at 128³ using FDK with the Hann window on an RTX 4070 SUPER. The stored
  field of view is 39 mm (0.3043 mm/voxel). Rendering flips the z array for an upright
  presentation, smooths with separable [1, 2, 1]/4, then shades the 0.30 isosurface
  of the normalized reconstruction. The 180 RGBA views are 512². The small scanner
  support visible below the walnut is retained. This is a CT surface, not a photo.
  The acquisition rig uses the same orthographic camera basis and scale, with
  a per-pixel depth buffer shared by the specimen, ray segments, detector triangles, source marker and volume boundary. The depth map is saved in `walnut_surface.npz`; trajectory examples are schematics.

To regenerate the surface, use the matching upstream 2.0.2 library/examples and
its measured `examples/data/walnut_cone.npz` in a CUDA-enabled environment:

```powershell
python examples/walnut_reconstruction.py --size 128 --algorithms --save-volume walnut128.npy
python docs/video/make_walnut_3d.py walnut128.npy
```

The surface script requires NumPy and PyTorch; it can also render on CPU. `make_inputs.py` can regenerate the measured and simulated inputs with CUDA; it does not regenerate the separately archived calibration history.

**Attribution:** Alexander Meaney (2022), *Cone-Beam Computed Tomography Dataset of
a Walnut*, https://doi.org/10.5281/zenodo.6986012, CC BY 4.0. `WALNUT_NOTICE.txt`
contains the upstream preprocessing and license notice. The film adds normalized
reconstruction displays and the surface-rendering transformations described above.


## Review refinements

- Chapter transitions fade the outgoing composition away before introducing the
  next one. Titles and formulas never overlap or splice together; the divider
  persists. Captions and replacement equations also exit before the next enters.
  Trajectory curves and source positions interpolate between scan types.
- Ordinary text uses Segoe UI with a consistent 20 / 24 / 28 / 36 point hierarchy
  and an 80 point wordmark. Lists are shaped as paragraphs with equal baseline
  spacing. Code uses Consolas at 22 points; ordinary labels do not use TeX.
- Projection, adjoint identity, loss, gradient and angular error use LaTeX
  with sans-serif math to fit the surrounding text.
- TV is labelled **TV-regularized**, with **Adam** on the optimizer/iteration line.
  All four methods share the same title color and typography. The comparison
  remains on screen through attribution and then transitions to the ending;
  there is no separate TV explanation or objective slide.
- README PNGs are regenerated from the archived arrays with
  `uv run --no-project --with matplotlib python docs/assets/make_reconstruction_figures.py`.
  Reconstruction data, PSNR values and grayscale limits are not changed.
- Depth regression checks:
  `python -m unittest discover -s docs/video -p test_scan_diagram.py`.
