Fan Beam Filtered Backprojection (FBP)
======================================

This example demonstrates 2D fan-beam filtered backprojection on a
Shepp-Logan phantom using the public call chain from ``diffct``::

    FanProjectorFunction.apply      -- differentiable fan forward projection
    parker_weights (optional)       -- short-scan redundancy weighting
    fan_cosine_weights              -- cos(gamma) pre-weighting
    ramp_filter_1d                  -- ramp filter along the detector axis
    angular_integration_weights     -- per-view integration weights
    fan_weighted_backproject        -- voxel-driven FBP backprojection gather

The accompanying source file is :file:`examples/fbp_fan.py`.

Overview
--------

Fan-beam FBP is the analytical reconstruction method for 2D fan-beam
CT on a circular source orbit. Compared with parallel beam it adds a
divergent-ray geometry (source-to-isocenter SID and source-to-detector
SDD distances), a cosine pre-weight, and a voxel-dependent distance
weight inside the backprojection. This example shows how to:

- Configure the 2D fan-beam geometry (source, detector, orbit).
- Generate fan-beam projections of a 2D Shepp-Logan phantom.
- Apply FBP cosine pre-weighting, ramp filtering (with optional
  zero-padding and frequency-domain windowing), and angular-integration
  weights.
- Run the dedicated voxel-driven FBP gather kernel through
  ``fan_weighted_backproject`` to reconstruct the image, already
  amplitude-calibrated.

Mathematical Background
-----------------------

**Fan-beam geometry**

- :math:`D` (``sid``): Source-to-Isocenter Distance. Source position at
  angle :math:`\beta` is
  :math:`\vec{r}_s(\beta) = (-D \sin\beta, D \cos\beta)`.
- :math:`D_{sd}` (``sdd``): Source-to-Detector Distance. The detector
  is at distance ``sdd`` from the source perpendicular to the central
  ray.
- :math:`u`: detector in-plane coordinate; :math:`du` is the detector
  cell pitch.
- :math:`\gamma = \arctan(u / D_{sd})`: fan angle of a detector cell.

**Forward projection**

The fan-beam projection at source angle :math:`\beta` and detector
position :math:`u` is

.. math::
   p(\beta, u) = \int_0^{\infty}
       f\!\left(\vec{r}_s(\beta) + t \cdot \vec{d}(\beta, u)\right) \, dt

where :math:`\vec{d}` is the unit ray direction from source to the
detector cell. ``FanProjectorFunction`` implements this integral via a
Siddon ray-march with bilinear interpolation.

**FBP algorithm**

1. **Cosine pre-weighting** of the raw projection:

   .. math::
      p_w(\beta, u) = \frac{D_{sd}}{\sqrt{D_{sd}^2 + u^2}}\,
                      p(\beta, u)
                   = p(\beta, u) \cos\gamma.

2. **Ramp filtering** along the detector axis:

   .. math::
      p_f(\beta, u) =
          \mathcal{F}_u^{-1}\bigl\{\,|\omega_u|\,
          \mathcal{F}_u\{p_w(\beta, u)\}\bigr\}.

   ``ramp_filter_1d`` lets you choose between Ram-Lak, Hann, Hamming,
   Cosine and Shepp-Logan shapes.

3. **Voxel-driven backprojection** with the classical FBP distance
   weight:

   .. math::
      f(x, y) = \frac{1}{2} \int_0^{2\pi}
          \frac{D^2}{U(\beta, x, y)^2}\,
              p_f\!\left(\beta, u_{xy}\right) d\beta,

   where
   :math:`U(\beta, x, y) = D + x \sin\beta - y \cos\beta` is the
   distance from the source to the pixel along the central ray
   direction, and the pixel's projected detector coordinate is

   .. math::
      u_{xy} = \frac{D_{sd}}{U}\,(x \cos\beta + y \sin\beta).

   ``fan_weighted_backproject`` dispatches to a dedicated voxel-driven
   gather kernel ``_fan_2d_fbp_backproject_kernel`` for this step. For
   each pixel it computes the projected :math:`u`, linearly samples
   the filtered sinogram, multiplies by :math:`(D/U)^2`, and
   accumulates across views. The Siddon-based autograd fan
   backprojector is reserved for the differentiable path used by
   iterative reconstruction and is **not** touched by this analytical
   pipeline.

**Ramp Filter Options**

``ramp_filter_1d`` accepts several high-precision parameters (all
optional and fully backward compatible):

- ``sample_spacing`` -- physical detector pitch along ``dim``. Set
  this to ``detector_spacing`` for fan FBP so the filter output is in
  physical units and the reconstructed amplitude is detector-pitch
  independent.
- ``pad_factor`` -- zero-pad the detector dimension to
  ``pad_factor * N`` before the FFT. ``2`` is recommended for FBP.
- ``window`` -- frequency-domain apodization. Options: ``None`` /
  ``"ram-lak"`` (unwindowed), ``"hann"`` (used in this example),
  ``"hamming"``, ``"cosine"``, ``"shepp-logan"``.
- ``use_rfft`` -- use ``torch.fft.rfft`` for real inputs
  (default ``True``, ``~2x`` faster than the complex path).

**Analytical FBP scale**

``fan_weighted_backproject`` multiplies its output by the analytical
constant ``sdd / (2 * pi * sid)``. The ``1/(2*pi)`` comes from the
Fourier convention used by ``ramp_filter_1d`` (the ``|omega|`` ramp
in radian frequency), and the ``sdd/sid`` factor corrects for the
fact that the ramp is applied on the physical detector plane rather
than the virtual isocenter-plane detector. The analytical cone FDK
helper uses the same constant for the same reason.

**Implementation Steps**

The ``main()`` function in :file:`examples/fbp_fan.py` follows the
same 8-step structure as the cone FDK example: volume geometry,
source trajectory, detector geometry, CUDA transfer, forward
projection, FBP pipeline (optional Parker / cosine pre-weight / ramp
filter / angular weights / voxel-gather backprojection / optional
ReLU), quantitative summary, visualization.

**Short-scan support (Parker weighting)**

For a short-scan trajectory (angular coverage of
:math:`\pi + 2\gamma_{\max}`), set ``apply_parker=True`` in the
example. The ``parker_weights`` helper builds the standard smooth
redundancy weight that tapers the two ends of the angular range.
When Parker is enabled the redundancy factor of the full-scan FBP
formula is dropped, so the example passes
``redundant_full_scan=False`` to ``angular_integration_weights``.

Source
------

.. literalinclude:: ../../examples/fbp_fan.py
   :language: python
   :linenos:
   :caption: 2D Fan Beam FBP Example
