API Reference
=============

This section provides documentation for the differentiable CT operators
and analytical reconstruction helpers in ``diffct``. Each autograd class
is a PyTorch ``torch.autograd.Function`` that runs a CUDA kernel under
the hood, enabling gradient propagation through the full projection /
backprojection pipeline. The analytical helpers (``ramp_filter_1d``,
``cone_cosine_weights``, ``cone_weighted_backproject`` and friends) are
intentionally kept as plain functions so they can be composed with the
autograd operators or used directly in one-shot FBP / FDK pipelines.

Overview
--------

The library exposes three families of operators, grouped by geometry:

- **Parallel beam (2D):** parallel-beam geometry for 2D reconstruction.
- **Fan beam (2D):** divergent fan-beam geometry.
- **Cone beam (3D):** full 3D cone-beam geometry for volumetric
  reconstruction with a 2D flat-panel detector.

Each geometry ships a differentiable forward projector and a
differentiable backprojector that are adjoints of each other (the
backprojector is the gradient of the projector). For cone-beam the
analytical FDK path is routed through a dedicated voxel-driven gather
kernel; see the ``cone_weighted_backproject`` notes below.

Parallel Beam Operators
-----------------------

The parallel beam geometry assumes parallel X-ray beams, commonly used
in synchrotron CT and some medical CT scanners.

.. autoclass:: diffct.differentiable.ParallelProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diffct.differentiable.ParallelBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:


Fan Beam Operators
------------------

Fan beam geometry uses a point X-ray source with a fan-shaped beam,
typical in medical CT scanners.

.. autoclass:: diffct.differentiable.FanProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diffct.differentiable.FanBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:


Cone Beam Operators
-------------------

Cone beam geometry extends fan beam to 3D with a cone-shaped X-ray beam
and a 2D detector. ``ConeProjectorFunction`` and
``ConeBackprojectorFunction`` are exact adjoints of each other (Siddon
traversal with trilinear interpolation) and drive the iterative cone
example. For analytical FDK, use ``cone_weighted_backproject`` — it
dispatches to a dedicated voxel-driven gather kernel with the classical
``(sid/U)^2`` distance weighting.

.. autoclass:: diffct.differentiable.ConeProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diffct.differentiable.ConeBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:


Analytical Reconstruction Helpers
---------------------------------

These helpers build the per-view pre-weights, angle-integration weights,
filter, and backprojection pieces of an analytical FBP/FDK pipeline.
They are plain functions (no autograd state) and can be freely mixed
with the autograd operators above.

.. autofunction:: diffct.differentiable.detector_coordinates_1d

.. autofunction:: diffct.differentiable.angular_integration_weights

.. autofunction:: diffct.differentiable.fan_cosine_weights

.. autofunction:: diffct.differentiable.cone_cosine_weights

.. autofunction:: diffct.differentiable.parker_weights

.. autofunction:: diffct.differentiable.ramp_filter_1d

.. autofunction:: diffct.differentiable.parallel_weighted_backproject

.. autofunction:: diffct.differentiable.fan_weighted_backproject

.. autofunction:: diffct.differentiable.cone_weighted_backproject


Ramp Filter Options
-------------------

``ramp_filter_1d`` is a generic 1D ramp filter used by every analytical
reconstruction example. Its call signature is::

    ramp_filter_1d(sinogram_tensor, dim=-1, sample_spacing=1.0,
                   pad_factor=1, window=None, use_rfft=True)

``sample_spacing``
    Physical detector-cell spacing along ``dim`` (e.g. ``du`` for the
    cone-beam case). The filter is rescaled by ``1 / sample_spacing``
    internally so the output is in physical units and the
    reconstruction amplitude stays calibrated when detector pitch
    changes. Pass ``1.0`` to reproduce the historical sample-unit
    behavior.

``pad_factor``
    Zero-pad the signal to ``pad_factor * N`` samples along ``dim``
    before the FFT. Common values:

    - ``1`` (default): no padding, fastest but prone to circular
      convolution wrap-around at the detector edges.
    - ``2``: recommended for cone-beam FDK, good trade-off.
    - ``4``: stricter padding for objects close to the detector edge.

``window``
    Frequency-domain apodization multiplied onto the bare Ram-Lak ramp:

    - ``None`` or ``"ram-lak"``: unwindowed Ram-Lak, sharpest, highest
      noise.
    - ``"hann"``: Hann window, strong high-frequency suppression.
    - ``"hamming"``: slightly milder than Hann.
    - ``"cosine"``: half-cosine rolloff.
    - ``"shepp-logan"``: ``sinc`` rolloff, classical choice.

``use_rfft``
    Use ``torch.fft.rfft`` / ``irfft`` when the input is real. Defaults
    to ``True``; set to ``False`` only if you need the complex FFT path.


Analytical FBP / FDK architecture
---------------------------------

Each of the three analytical backprojection helpers dispatches to a
dedicated voxel-driven gather kernel (parallel / fan / cone), separate
from the Siddon-based ray-driven scatter kernels that drive autograd.
The autograd kernels are the pure adjoints ``P^T`` of the forward
projectors (no distance weighting, no magnification scale), so the
autograd classes form self-consistent forward/backward adjoint pairs.
The analytical helpers on top apply the appropriate FBP/FDK distance
weighting and the analytical scale so the returned image is already
amplitude-calibrated.

Scale factors applied inside the analytical helpers:

- ``parallel_weighted_backproject``: multiplies by ``1 / (2 * pi)``
  (Fourier-convention constant; parallel beam has no source and no
  magnification).
- ``fan_weighted_backproject``: multiplies by ``sdd / (2 * pi * sid)``
  (Fourier constant times physical-detector-to-isocenter-plane
  magnification).
- ``cone_weighted_backproject``: multiplies by ``sdd / (2 * pi * sid)``
  (same derivation as the fan case; the extra third dimension does
  not change the ramp-filter convention).

All three helpers use ``(sid / U)^2`` as the FDK/FBP distance weight
for the divergent-beam cases (fan and cone), where
``U = sid + x*sin(phi) - y*cos(phi)`` is the distance from the source
to the pixel/voxel along the central ray direction. Parallel beam has
no distance weighting.


Usage Notes
-----------

**Memory Management**

- All operators work on CUDA tensors for optimal performance.
- Ensure sufficient GPU memory for the chosen volume / sinogram size.
- Use ``torch.cuda.empty_cache()`` to release cached allocations when
  switching to a large job.

**Gradient Computation**

- All autograd operators support automatic differentiation.
- Gradients flow through both forward and adjoint paths.
- Set ``requires_grad=True`` on input tensors to enable gradients.
- Analytical helpers (``ramp_filter_1d``, ``angular_integration_weights``,
  ``cone_weighted_backproject``, ...) are also differentiable in the
  torch sense (they are pure tensor ops / autograd.Functions), so you
  can use them inside a loss.

**Performance Considerations**

- Pass contiguous tensors with the expected dimension order
  (``(D, H, W)`` for 3D volumes, ``(num_views, det_u, det_v)`` for cone
  sinograms). The library validates layout and will raise if a
  non-contiguous or transposed tensor is passed.
- Use ``pad_factor=2`` in ``ramp_filter_1d`` for cone-beam FDK; the
  extra FFT cost is usually negligible compared to the backprojection.

**Coordinate Systems**

- Image / volume coordinates: integer indices, ``(0, 0, 0)`` at the
  corner of the array.
- Detector coordinates: centered at the detector array center, with
  ``u`` in-plane and ``v`` axial for cone beam.
- Rotation: counter-clockwise around the z-axis (right-hand rule).
