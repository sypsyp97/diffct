# diffct introduction

The film preserves the upstream dark, serif presentation: LaTeX typography,
numbered chapters, a thin progress rule, framed CT images and monospace code
cards. The revised palette is blue-black, warm ivory, cyan and amber, with muted
blue/lavender/sage for trajectory and GPU identification. The measured walnut is
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

The GPU chapter also uses computed 2D illustration inputs. To regenerate them
with the repository's matching PyTorch environment:

```powershell
python docs/video/make_gpu_example.py
python -m unittest discover -s docs/video -p test_gpu_example.py
```

The saved `gpu_example.npz` lets the film render without PyTorch or a GPU.

## Story and coverage

1. **Opening:** rotating 3D walnut, project name, PyTorch and feature overview.
2. **Trajectories:** circular, helical, saddle, sinusoidal and discrete per-view
   poses; a source, detector and cone rays orbit a 3D specimen/volume boundary.
   The same `Projector` interface takes each trajectory.
3. **Autograd:** walnut slice and sinogram, projection and matched adjoint,
   inner-product identity, loss/backward code, volume/sinogram second derivatives.
4. **Geometry gradients:** a clearly labelled pose-correction schematic alongside
   the recorded calibration loss, angle error and first-order geometry scope.
5. **Multi-GPU:** one walnut slice stays on the left and its simulated sinogram
   on the right. Four numbered image tiles move through one GPU buffer; their
   computed projections accumulate in the CPU output, then the next view batches
   fill the rest. Two GPUs first receive copies of the whole image and compute
   separate view ranges (`views`); their result rows stack. Next they receive
   different image halves (`space`); their overlapping projections add. Four GPUs
   combine both splits (`hybrid`, grid `(2, 2)`): columns own view groups and rows
   own image halves. Backprojection reverses the visible data flow and adds
   view-group contributions to each image half. The result is labelled unfiltered
   backprojection (matched adjoint); its caption explains that spreading values
   along rays produces blur and a raised background, rather than a reconstruction.
   The final chart retains explicitly archived view-partition timings, which do
   not measure the hybrid backend.
6. **Measured reconstruction:** four real projections, then FDK / SIRT / CGLS /
   TV as a four-column, two-row axial/coronal comparison with iteration counts.
7. **Ending:** rotating walnut, plain installation for CPU/PyTorch ROCm and CUDA
   extras for native NVIDIA execution, the matching PyTorch prerequisite,
   documentation and repository links.

The first shot shows **GPU buffer residency**. The later worker cards show
logical data per **rank**, which may themselves be streamed in tiles as in the
first shot. `views` still replicates the rank volume; `space` and `hybrid` own
only first-axis slabs. Caller-owned CUDA inputs/results and autograd graph
storage are outside the bounded streaming buffers. The complete sinogram or
adjoint at the side is explicitly **assembled on CPU for this illustration**;
distributed projection normally returns owned view rows replicated within each
space group, while the adjoint returns an owned image slab replicated across
view groups. The clip does not imply an automatic global gather.

GPU cards are a data-flow schematic, not measured execution time or memory use.
The distributed API excerpts assume one process per GPU, local device binding,
and an initialized process group. Hybrid selects the PyTorch backend automatically.
All ranks must use consistent shapes, partitions and chunk hints. The loss formula in the
autograd chapter is the ordinary single-process loss; distributed replica
divisors are documented in [EXECUTION.md](../EXECUTION.md#hybrid-ownership-and-losses).
Code cards are API excerpts: `...` stands for the acquisition/shape arguments.
Set geometry `requires_grad` before constructing the projector.

## Content cross-check

Every visible caption, code card, formula, label and plotted value was checked
within these shot groups. Schematics explain API behavior; archived plots and
reconstruction panels retain their recorded inputs.

| Shot | Content checked | Evidence |
|---|---|---|
| Opening | Title, five chapters, measured CT surface | `walnut_surface.npz` |
| Trajectories | Five schematic paths; source/centre/u/v tuple; valid per-view poses | [Projector](../../diffct/operators.py), unchanged [ScanDiagram](scan_diagram.py) |
| Autograd | Simulated parallel sinogram; Siddon path sum; matched adjoint; ordinary squared-loss gradient; batch/second data derivatives | [Projector](../../diffct/operators.py), [portable traversal](../../diffct/backends.py), `make_inputs.py`, `data2d.npz` |
| Geometry | First-order source/centre/axis gradients; schematic pose dots; log10 loss, 150 steps, angle RMSE 0.55° to 0.006° | [geometry VJP](../../diffct/kernels/geometry_vjp.py), `calib_history.json` |
| Streaming | Four actual image tiles, one GPU buffer; first 16 views accumulate tile contributions, then repeat for all 64 views | `gpu_example.npz`, [generator](make_gpu_example.py), [Projector streamed execution](../../diffct/operators.py) |
| Views / space | Same image copies with different view rows stack; different image halves with the same views add | [Partition](../../diffct/distribution.py), computed `tile_projection` |
| Hybrid | `(2, 2)` ordering: GPUs 0/1 own views 1–32, GPUs 2/3 views 33–64; 0/2 own the first image half, 1/3 the second; projection adds vertically, adjoint horizontally | [Partition](../../diffct/distribution.py), [portable reductions](../../diffct/backends.py), computed `view_backprojection` |
| Scaling | Archived 2.0.2 views timings: 30.08 / 8.08 / 5.59 ms, calculated 3.7× / 5.4×; 128³, 360 views, A100 | `docs/video/scaling.json`; separate from the 2.2.0 README chart |
| Measured data | Four transposed measured projections; existing axial/coronal orientation, common [0, 1] range; FDK / SIRT 200 / CGLS 20 / Adam TV 300; dataset attribution | Unchanged `measured()` and `walnut_measured.npz`, `make_inputs.py`, `WALNUT_NOTICE.txt` |
| Ending | CPU/PyTorch ROCm installation, native CUDA 12/13 extras, matching PyTorch prerequisite and project URLs | [Install guide](../../README.md#install), `pyproject.toml`, [backend selection](../../diffct/backends.py) |

## Render review (2026-10-15)

The exported film is 151.367 s: **4,541 MP4 frames**, 1920 × 1080 at constant
30 fps. A Cairo render observer recorded visible Tex/MathTex identities and
glyph bounds for every written frame, including 1,272 repeated wait frames.
Complete decoding confirms source/trace/export counts agree. No text violates
the 0.1-scene-unit viewport margin; no distinct text bounding boxes overlap.

The 800 × 450 GIF decodes into **1,130 frames**, covering all **1,211** 8 fps source
samples; identical samples merge. Its 151.37 s duration differs by less than one
sampling interval. All frames share one 255-colour palette, and pixels unchanged
from the previous frame are stored as transparent, so the file is 12.1 MB (41.2 MB
with a separate 64-colour palette per frame). Against the MP4 at the same sample
times, every displayed GIF frame reaches at least 39.9 dB PSNR (median 42.2 dB).
To rebuild only the GIF from the committed MP4, run
`python docs/video/render.py --gif-only`. Two MP4 frames (9.100 s and 141.967 s)
are background-only at sequential chapter fade midpoints; these are the retained
transition style.

Manual pixel review covers text compositions at fade/solid states, animation
midpoints, chapter transitions, regular animation samples and full-resolution
keyframes. The unfiltered-backprojection label and explanation were also checked
in a preview, including their fade and five-second reading hold. Moving packets
pass behind protected GPU identifiers and operation labels; label changes fade
out before the replacement fades in. Bounds/provenance checks do not prove
semantics or occlusion; the content table above and pixel review provide those
checks. The GPU layout uses mirrored horizontal connectors, shared title and
description baselines, and input/output centres aligned with each GPU card group.
Image aspect ratios are preserved. The tile, operation and GPU labels remain
separate as data moves through the cards. Local full-story review sheets remain
under ignored `.validation/video-intuitive/final-v4/`; the latest render and GPU
alignment checks are under `.validation/video-alignment/final/`.

The GPU illustration numeric checks passed (7 tests), including regeneration of
the saved fixture. Existing depth checks passed (5 tests). Main's measured input
arrays and display transforms are unchanged. Python compilation and
`git diff --check` passed. No new ROCm/NCCL/cross-node performance measurements
were made for this video.

## Data and provenance

The API/ownership story is checked against released main
`db2daa954012500c05549deb833206a83bc10f44` (2.3.0). The video source and existing
orientation/depth fixes start from that commit. This PR changes no library code,
measured arrays, slice transforms or scan-diagram geometry.

Archived reconstruction/calibration inputs originate from
`50a897f8860346ef906d4c909db081452c411657` (2.0.2); these are retained results,
not newly measured 2.3.0 solver/accuracy results.

- `data2d.npz` is byte-identical to that commit. `walnut_measured.npz` retains
  the same recorded values with main's coronal display correction (`c6d6ced`):
  its four coronal fields are vertical flips of the archived arrays, while the
  four axial fields and projections match exactly. Main's `4ac2e15` additionally
  displays projections as `p.T` without an extra vertical flip. Both corrections
  are retained. The file contains four measured cone-beam projections plus FDK, SIRT (200),
  CGLS (20), and TV (300, weight 0.3) axial/coronal slices at 256³. All reconstruction
  panels use [0, 1]. `data2d.npz` contains a walnut slice and a SciPy rotate/sum
  **display simulation** of a parallel sinogram; the film labels it accordingly.
- `gpu_example.npz` is a separate **computed 2D data-flow illustration**, generated
  by [make_gpu_example.py](make_gpu_example.py) with the released library's CPU
  PyTorch float64 Siddon projector. The measured 256² axial slice is mean-pooled
  by 4 × 4 to 64². Geometry has 64 parallel views on [0, π), 96 detector bins,
  unit voxel/detector spacing. Four 16-row contributions keep global coordinates;
  pairs form the two 32-row spatial owners. View groups contain 32 rows each.
  `tile_projection` sums to the full sinogram, and `view_backprojection` sums to
  its matched adjoint. All projected panels share the full sinogram's maximum;
  all adjoint panels share the complete adjoint's maximum. In this example, the
  unfiltered adjoint is blurred with a raised background because ray values
  spread across the image. The video labels it as unfiltered backprojection and
  explains its appearance. The seven numeric checks include independent
  axis-aligned row/column path sums and regeneration of the
  committed fixture. This introduces no new acquisition, solver or GPU benchmark.
- `calib_history.json` is the saved, separate 64³ calibration run. Its seven
  samples are plotted on a labelled log10 axis, with steps 0–149 (150 steps).
  Pose dots are a schematic, not recovered measurements or saved optimizer poses.
  Displayed angle errors come directly from this JSON. Geometry gradients are
  first order; volume/sinogram gradients also support second derivatives.
- `docs/video/scaling.json` retains the archived CGLS 128³/360-view timings from
  2.0.2 and calculates speedup from them. Hardware: A100-SXM-64GB. The film labels
  the version and `views` partition explicitly. `docs/assets/scaling.json` holds
  the separate 2.2.0/512³ README benchmark; it is not the film's input. Neither
  benchmark measures portable/hybrid execution in 2.3.0.
- `walnut_surface.npz` is derived from the **real measured dataset**, not
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
  next one. Titles and formulas never overlap or splice together; the progress
  track persists. Captions and replacement equations also exit before the next enters.
  Trajectory curves and source positions interpolate between scan types.
- Text returns to upstream's `Tex` and `MathTex` rendering rather than Pango
  `Text`. The original Latin Modern serif/monospace family handles kerning and
  ligatures in LaTeX. Body text and headings use the original 20 / 24 / 28 / 36
  scale (25 / 30 / 35 / 45 TeX points); compact GPU annotations use 17–22.
  List and caption rows share baselines using the upstream baseline helper. Code spaces are
  explicitly preserved in TeX. No per-character tracking or system font fallback
  is used. This also avoids the old small-size Pango spacing path documented in
  [Manim's 0.22 release notes](https://docs.manim.community/en/stable/changelog/0.22.0-changelog.html).
- Opening and ending share a centered title, subtitle and walnut, with identical
  positions and type sizes. The opening chapter map uses the upstream centered
  three-plus-two arrangement. Trajectory content follows equal-width columns.
  GPU inputs and assembled outputs keep fixed positions as the division of work
  changes. GPU cards contain actual walnut regions and computed sinograms rather
  than anonymous blocks. Forward contributions visibly accumulate; the hybrid
  adjoint moves data in the reverse direction and adds across the view groups.
- The opening restores the five-chapter overview, the scan rig keeps an amber
  source marker, and trajectory/GPU colors identify the active path or partition.
  Projection and adjoint use cyan and amber with a shared serif formula style.
- TV is labelled **TV-regularized**, with **Adam** on the optimizer/iteration line.
  All four methods share the same title color and typography. The comparison
  remains on screen through attribution and then transitions to the ending;
  there is no separate TV explanation or objective slide.
- README PNGs are regenerated from the archived arrays with
  `uv run --no-project --with matplotlib python docs/assets/make_reconstruction_figures.py`.
  Reconstruction data, PSNR values and grayscale limits are not changed.
- Depth regression checks:
  `python -m unittest discover -s docs/video -p test_scan_diagram.py`.
