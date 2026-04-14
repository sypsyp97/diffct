Parallel Beam Filtered Backprojection (FBP)
===========================================

This example demonstrates 2D parallel-beam filtered backprojection
on a Shepp-Logan phantom using the public call chain from ``diffct``::

    ParallelProjectorFunction.apply  -- differentiable parallel forward
    ramp_filter_1d                   -- ramp filter along the detector axis
    angular_integration_weights      -- per-view integration weights
    parallel_weighted_backproject    -- voxel-driven FBP backprojection gather

The accompanying source file is :file:`examples/fbp_parallel.py`.

Overview
--------

Parallel-beam FBP is the classical analytical reconstruction method
for 2D parallel CT. Because there is no source (rays are collimated),
the pipeline is the simplest of the three analytical paths in
``diffct``: no cosine pre-weight, no distance weight, no magnification
correction. What remains is ramp filtering, a per-view angular
integration weight, and a ``1/(2*pi)`` analytical constant that the
backprojection helper applies internally.

This example shows how to:

- Sample a circular full-scan trajectory and a flat detector.
- Generate parallel projections of a 2D Shepp-Logan phantom.
- Apply ramp filtering with zero-padding and a Hann window.
- Run the dedicated voxel-driven FBP gather kernel through
  ``parallel_weighted_backproject`` to reconstruct the image, already
  amplitude-calibrated.

Mathematical Background
-----------------------

**Parallel-beam geometry**

Rays at angle :math:`\theta` are collimated in the direction
:math:`(\cos\theta, \sin\theta)`. A detector cell at position
:math:`u` on the detector axis :math:`(-\sin\theta, \cos\theta)`
samples a ray passing through the point
:math:`(-u \sin\theta,\; u \cos\theta)` in image-space coordinates
centered on the isocenter. The corresponding line integral is the
classical Radon transform

.. math::
   p(\theta, u) = \int_{-\infty}^{\infty}
       f\!\left(-u \sin\theta + s \cos\theta,\;
                 u \cos\theta + s \sin\theta\right) ds.

The pixel :math:`(x, y)` projects to the detector at
:math:`u_{xy} = -x \sin\theta + y \cos\theta`.

**FBP algorithm**

1. **Ramp filtering** along the detector axis:

   .. math::
      p_f(\theta, u) =
          \mathcal{F}_u^{-1}\bigl\{\,|\omega_u|\,
          \mathcal{F}_u\{p(\theta, u)\}\bigr\}.

2. **Voxel-driven backprojection** without any distance weighting:

   .. math::
      f(x, y) = \frac{1}{4\pi}
          \int_0^{2\pi}
              p_f\!\left(\theta, u_{xy}(\theta)\right) d\theta.

   The ``1/(4\pi) \cdot \int_0^{2\pi}`` factor is equivalent to
   :math:`\frac{1}{2\pi} \cdot \int_0^{\pi}`; ``diffct`` uses the
   full :math:`[0, 2\pi)` form with ``redundant_full_scan=True`` in
   ``angular_integration_weights`` to absorb the factor of 1/2.

   ``parallel_weighted_backproject`` dispatches to a dedicated
   voxel-driven gather kernel ``_parallel_2d_fbp_backproject_kernel``
   for this step. For each pixel and each view it computes
   :math:`u_{xy}`, linearly samples the filtered sinogram, and sums
   - no distance weighting, no magnification. The Siddon-based
   autograd parallel backprojector is reserved for the differentiable
   path used by iterative reconstruction and is *not* touched by this
   analytical pipeline.

**Ramp Filter Options**

``ramp_filter_1d`` accepts the same high-precision optional kwargs as
in the fan and cone examples: ``sample_spacing`` (set to
``detector_spacing``), ``pad_factor`` (``2`` recommended), ``window``
(``None`` / ``"ram-lak"``, ``"hann"``, ``"hamming"``, ``"cosine"``,
``"shepp-logan"``), and ``use_rfft``. The call is backward compatible
so passing only ``(sino, dim=1)`` still works.

**Analytical FBP scale**

``parallel_weighted_backproject`` multiplies its output by
``1 / (2 * pi)``. This is the Fourier-convention constant that
matches the ``|omega|`` ramp used by ``ramp_filter_1d``. Unlike the
fan and cone helpers there is no ``sdd/sid`` magnification factor
because parallel beam has no source.

**Implementation Steps**

The ``main()`` function in :file:`examples/fbp_parallel.py` follows
the unified 8-step structure: volume geometry, source trajectory,
detector geometry, CUDA transfer, forward projection, FBP pipeline
(ramp filter / angular weights / voxel-gather backprojection /
optional ReLU), quantitative summary (including a gradient sanity
check), visualization.

Source
------

.. literalinclude:: ../../examples/fbp_parallel.py
   :language: python
   :linenos:
   :caption: 2D Parallel Beam FBP Example
