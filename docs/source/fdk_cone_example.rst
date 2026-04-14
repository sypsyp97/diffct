Cone Beam FDK Reconstruction
============================

This example demonstrates 3D cone-beam FDK (Feldkamp-Davis-Kress)
reconstruction on a 3D Shepp-Logan phantom using the public call chain
from ``diffct``::

    ConeProjectorFunction.apply    -- differentiable cone forward projection
    cone_cosine_weights            -- 1/r^2 cosine pre-weighting
    ramp_filter_1d                 -- row-wise ramp filter along detector-u
    angular_integration_weights    -- per-view integration weights
    cone_weighted_backproject      -- voxel-driven FDK backprojection gather

The accompanying source file is :file:`examples/fdk_cone.py`.

Overview
--------

The FDK algorithm is the standard analytical method for 3D cone-beam
CT reconstruction on a circular source orbit. This example shows how
to:

- Configure the 3D cone-beam geometry (source, detector, orbit).
- Generate cone-beam projections of a 3D Shepp-Logan phantom.
- Apply FDK cosine pre-weighting, ramp filtering (with optional
  zero-padding and frequency-domain windowing), and angular-integration
  weights.
- Run the dedicated voxel-driven FDK gather kernel through
  ``cone_weighted_backproject`` to reconstruct the volume, already
  amplitude-calibrated.

Mathematical Background
-----------------------

**Cone-Beam Geometry**

Cone-beam CT uses a point X-ray source and a 2D detector. Key
parameters:

- :math:`D` (``sid``): Source-to-Isocenter Distance. Source position at
  angle :math:`\phi` is
  :math:`\vec{r}_s(\phi) = (-D \sin\phi, D \cos\phi, 0)`.
- :math:`D_{sd}` (``sdd``): Source-to-Detector Distance. The detector
  plane is perpendicular to the central ray, ``sdd`` away from the
  source. The magnification at isocenter is :math:`D_{sd}/D`.
- :math:`(u, v)`: detector in-plane and axial coordinates.
- :math:`(du, dv)`: detector cell pitch.

**3D Forward Projection**

The cone-beam projection at source angle :math:`\phi` and detector
position :math:`(u, v)` is

.. math::
   p(\phi, u, v) = \int_0^{\infty}
       f\!\left(\vec{r}_s(\phi) + t \cdot \vec{d}(\phi, u, v)\right) \, dt

where :math:`\vec{d}` is the unit ray direction from source to
detector cell. ``ConeProjectorFunction`` implements this integral via
a Siddon ray-march with trilinear interpolation.

**FDK Algorithm**

The Feldkamp-Davis-Kress formula reconstructs an approximation of the
3D volume in three steps:

1. **Cosine pre-weighting** of the raw projection:

   .. math::
      p_w(\phi, u, v) = \frac{D_{sd}}{\sqrt{D_{sd}^2 + u^2 + v^2}}\,
                       p(\phi, u, v).

2. **Row-wise ramp filtering** along the detector-u direction:

   .. math::
      p_f(\phi, u, v) =
          \mathcal{F}_u^{-1}\bigl\{\,|\omega_u|\,
          \mathcal{F}_u\{p_w(\phi, u, v)\}\bigr\}.

   Each detector row is filtered independently. ``ramp_filter_1d``
   lets you choose between Ram-Lak, Hann, Hamming, Cosine and
   Shepp-Logan shapes (see below).

3. **Voxel-driven backprojection** with the classical FDK distance
   weight:

   .. math::
      f(x, y, z) = \frac{1}{2}
          \int_0^{2\pi} \frac{D^2}{U(\phi, x, y)^2}\,
              p_f\!\left(\phi, u_{xyz}, v_{xyz}\right) d\phi,

   where
   :math:`U(\phi, x, y) = D + x \sin\phi - y \cos\phi` is the distance
   from the source to the voxel along the central ray direction, and
   the voxel's projected detector coordinates are

   .. math::
      u_{xyz} = \frac{D_{sd}}{U}\,(x \cos\phi + y \sin\phi),
      \qquad
      v_{xyz} = \frac{D_{sd}}{U}\, z.

   ``cone_weighted_backproject`` dispatches to a dedicated
   voxel-driven gather kernel for this step. For every voxel it
   computes the projected :math:`(u, v)`, bilinearly samples the
   filtered sinogram, multiplies by :math:`(D/U)^2`, and accumulates
   across views. The Siddon-based autograd cone backprojector is
   reserved for the differentiable path used by iterative
   reconstruction and is not touched by this analytical pipeline.

**Implementation Steps**

The ``main()`` function in :file:`examples/fdk_cone.py` goes through
eight stages:

1. **Volume geometry** -- set ``Nx``, ``Ny``, ``Nz``, ``voxel_spacing``.
2. **Source trajectory** -- sample ``num_views`` angles on :math:`[0, 2\pi)`.
3. **Detector geometry** -- set ``det_u``, ``det_v``, ``du``, ``dv``,
   ``detector_offset_u``, ``detector_offset_v``, ``sdd``, ``sid``.
4. **CUDA transfer** -- move phantom and angles to the GPU.
5. **Forward projection** via ``ConeProjectorFunction.apply``.
6. **FDK pipeline**:

   a. ``cone_cosine_weights`` -> pre-weighted sinogram.
   b. ``ramp_filter_1d`` with ``sample_spacing=du``, ``pad_factor=2``,
      ``window="hann"``.
   c. ``angular_integration_weights`` with ``redundant_full_scan=True``.
   d. ``cone_weighted_backproject`` -> raw FDK volume.
   e. ``F.relu`` -> optional non-negativity clamp.

7. **Quantitative summary** -- print raw MSE, clamped MSE, raw / clamped
   reconstruction ranges.
8. **Visualization** -- mid-slice phantom, mid-view sinogram, mid-slice
   reconstruction.

**Ramp Filter Options**

``ramp_filter_1d`` accepts several high-precision parameters (all
optional and fully backward compatible with the historical
``ramp_filter_1d(sino, dim=1)`` call):

- ``sample_spacing`` -- physical detector pitch along ``dim``. Set
  this to ``du`` for cone-beam FDK so the filter output is in
  physical units and the reconstructed amplitude is detector-pitch
  independent.
- ``pad_factor`` -- zero-pad the detector dimension to
  ``pad_factor * N`` before the FFT. ``2`` is recommended for FDK.
- ``window`` -- frequency-domain apodization. Options:
  ``None`` / ``"ram-lak"`` (unwindowed), ``"hann"`` (used in this
  example), ``"hamming"``, ``"cosine"``, ``"shepp-logan"``.
- ``use_rfft`` -- use ``torch.fft.rfft`` for real inputs
  (default ``True``, ``~2x`` faster than the complex path).

**3D Shepp-Logan Phantom**

The 3D phantom extends the 2D version with 10 ellipsoids:

- A large outer skull ellipsoid.
- Mid-sized ellipsoids for brain tissue contrast.
- Small ellipsoids representing ventricles and high-contrast lesions.

Each ellipsoid is described by a center :math:`(x_0, y_0, z_0)`,
semi-axes :math:`(a, b, c)`, a rotation angle :math:`\phi` around the
z-axis, and an attenuation amplitude.

**FDK Approximations and Limitations**

FDK uses a few approximations on circular orbits:

- **Circular source trajectory** (no helical motion).
- **Row-wise 1D ramp filtering** rather than a 2D filter.
- **Small-cone approximation** -- reconstruction quality degrades as
  the cone angle grows.

These introduce cone-beam artifacts at large cone angles, but FDK
remains the default analytical method due to its simplicity and
efficiency.

Source
------

.. literalinclude:: ../../examples/fdk_cone.py
   :language: python
   :linenos:
   :caption: 3D Cone Beam FDK Example
