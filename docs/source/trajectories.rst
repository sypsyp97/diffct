Trajectories and Geometry
=========================

Trajectory format
-----------------

``Projector`` takes one row per view in each trajectory tensor. The beam type
selects the tuple layout. The helpers in ``diffct`` create these tensors. You can
also pass measured or calibrated poses in the same layout.

.. list-table::
   :header-rows: 1

   * - Beam
     - Tuple (each component's shape)
     - Volume / sinogram
   * - ``parallel``
     - ``(ray_dir, det_origin, det_u)``, each ``(N, 2)``
     - ``(H, W)`` / ``(N, U)``
   * - ``fan``
     - ``(src_pos, det_center, det_u)``, each ``(N, 2)``
     - ``(H, W)`` / ``(N, U)``
   * - ``cone`` (default)
     - ``(src_pos, det_center, det_u, det_v)``, each ``(N, 3)``
     - ``(D, H, W)`` / ``(N, U, V)``

Coordinates are Cartesian ``(x, y[, z])``. Volume indexing is ``[y, x]`` for 2D
and ``[z, y, x]`` for 3D. The origin is the volume centre. Use one length unit
for positions, voxel spacing and detector spacing. Voxel spacing is a single
positive scalar.

Detector axes are unit direction vectors, not pixel-step vectors. The detector
spacing sets the pixel pitch. A scalar spacing applies to both cone detector axes.
With the default flat detector, pixel ``(u, v)`` is at:

.. code-block:: text

   det_center + (u - (U - 1) / 2) * du * det_u
              + (v - (V - 1) / 2) * dv * det_v

A measured sinogram in ``(views, V, U)`` order must be transposed to
``(views, U, V)`` before use. Check the physical direction of each detector axis.
Reshaping does not correct axis orientation.

Custom acquisition example
--------------------------

This example modifies a generated helix with a per-view detector shift. For a
measured acquisition, replace the four tensors with your calibrated arrays in the
coordinate system above.

.. code-block:: python

   import torch
   from diffct import Projector, spiral_trajectory_3d

   trajectory = spiral_trajectory_3d(
       90, sid=80.0, sdd=128.0, z_range=10.0, n_turns=1.0, device="cpu")
   source, center, u_axis, v_axis = trajectory
   phase = torch.arange(90, dtype=torch.float32) * (2 * torch.pi / 90)
   center = center + (0.25 * phase.sin())[:, None] * u_axis
   A = Projector((source, center, u_axis, v_axis), (32, 32, 32), (96, 64),
                 detector_spacing=(0.8, 0.8))
   volume = torch.ones(A.volume_shape, device="cuda")
   sinogram = A.project(volume)  # (90, 96, 64)

All components must be finite floating-point tensors with the same nonzero view
count. ``ray_dir`` and ``det_u`` must be orthogonal for parallel beams. ``det_u``
and ``det_v`` must be orthogonal for cone beams. With flat fan/cone detectors,
the source must not coincide with the detector centre or view the detector
edge-on. The nearer of source and detector centre must lie within 1e6 voxel
spacings of the origin. Surface detectors instead validate each actual
source/pixel pair: the endpoints must remain distinct in float32, and the
nearer endpoint of every ray must lie within 1e6 voxel spacings of the origin.
Both endpoints must lie within 1e15 voxel spacings.

Arbitrary trajectories do not guarantee enough angular coverage for an inverse
problem. Use iterative methods with regularization for sparse or incomplete
acquisitions. The analytical helpers assume a circular orbit (see :doc:`api`).

.. _detector-surfaces:

Parameterized detector surfaces
-------------------------------

``Projector(..., detector_surface=surface)`` places each pixel
on a native curve. The legacy two-argument callback places pixels
on an arc, cylinder or another parameterized surface. For cone beams it receives
physical float64 ``u, v`` grids on CPU with shape ``(U, V)`` and ``ij`` indexing. It returns local
``(u, v, n)`` offsets shaped ``(U, V, 3)`` or ``(views, U, V, 3)``. Here ``views``
is the total trajectory view count, also in distributed mode. The world point is:

.. code-block:: text

   center + offset_u * det_u + offset_v * det_v
          + offset_n * cross(det_u, det_v)

For 2D beams, ``u, v`` have shape ``(U,)`` and ``v`` is zero. Return ``(U, 3)``
or ``(views, U, 3)`` with zero middle offsets; the normal is
``(det_u_y, -det_u_x)``. A cylindrical surface can be written as:

.. code-block:: python

   radius = torch.tensor(128.0, requires_grad=True)

   def cylinder(u, v):
       u, v = u.to(radius.device), v.to(radius.device)
       angle = u / radius
       return torch.stack((radius * angle.sin(), v,
                           radius * (angle.cos() - 1)), dim=-1)

   C = Projector(trajectory, (32, 32, 32), (96, 64),
                 detector_spacing=(0.8, 1.0), detector_surface=cylinder)

For this generated frame the normal points away from the source; negative
normal offsets and a radius equal to ``sdd`` centre the cylinder on the source.

The callback is sampled at construction and afresh on each operation, and its
captured parameters receive first-order gradients. Every sample must be a finite floating-point tensor of
the documented shape; fan/cone pixels cannot coincide with the source.
``project`` and ``backproject`` retain their matched Siddon model, with one ray
per pixel and no detector-area integration. They operate directly on the native
curved grid without resampling. Run ``python examples/curved_detector.py`` for a
complete circular cone example, also shown in :doc:`examples`.

Legacy streamed operations require CPU offsets and retain complete host
callback geometry. CUDA offsets remain supported when
the automatic CUDA full-volume path fits. See :doc:`chunking` for large-volume
execution and memory requirements.

For bounded view/pixel sampling, pass explicit tensor parameters:

.. code-block:: python

   from diffct import ParameterizedSurface

   def sample(u, v, view_indices, r):
       return torch.stack((r * torch.sin(u / r), v,
                           r * (torch.cos(u / r) - 1)), dim=-1)

   surface = ParameterizedSurface(sample, parameters=(radius,))
   C = Projector(trajectory, (32, 32, 32), (96, 64),
                 detector_spacing=(0.8, 1.0), detector_surface=surface,
                 view_chunk_size=8, detector_chunk_shape=(32, 32))

``u,v`` are a rectangle of global physical pixel centres; ``view_indices``
contains global view IDs. Return ``(*pixel_shape,3)`` shared offsets or
``(len(view_indices),*pixel_shape,3)`` batched offsets. In 2D the middle
component remains zero. Streamed grids, IDs and explicit parameter snapshots
are CPU tensors; parameter dtypes and original gradient devices are preserved.
Offsets and world points are validated on every executing batch.

The sampler is pure. Mutable geometry tensors must be explicit parameters;
other captured configuration must be immutable. Coupling across views uses
global IDs or explicit parameters, including values for omitted views. Each
forward captures the sampler and one set of frame/parameter values. Backward
recomputes local geometry from those snapshots and reduces pixel cotangents to
parameter gradients before communication. It retains first-order geometry
gradients and data Hessians. Explicit parameter snapshots scale with parameter
count; allocations inside user samplers remain caller-controlled. Do not cache
mutable callback values across calls.

Analytical FBP/FDK and weighting helpers continue to assume flat detectors and
do not accept the callback. Reusing a flat-detector FDK requires matching-ray
resampling of the line integrals onto a covered virtual flat grid; this adds
interpolation error and is outside the core operator.

Geometry gradients
------------------

Gradients with respect to trajectory tensors are first order only. Set
``requires_grad=True`` on each tensor before you construct ``Projector``. The
projector keeps references to all trajectory components when any component
requires gradients, and reads their current values on each call. If none require
gradients, it clones the entire trajectory. For example:

.. code-block:: python

   # Continue from the custom acquisition above.
   target = sinogram.detach()
   learnable_center = center.clone().requires_grad_()
   B = Projector((source, learnable_center, u_axis, v_axis), (32, 32, 32),
                 (96, 64), detector_spacing=0.8)
   loss = (B.project(volume) - target).square().sum()
   loss.backward()
   print(learnable_center.grad.shape)  # (90, 3)

This loss uses the same target geometry, so the gradient is zero. For a real
calibration loop, see ``examples/geometry_calibration.py``.

If poses come from learnable angles or offsets, rebuild the derived tensors and
the ``Projector`` inside each optimization step. Parameterize rotations so that
axes stay unit length and orthogonal. Trajectory frame checks run only at
construction; sampled surface positions are also checked on every operation.

Limits:

- Second derivatives with respect to geometry raise an error.
- Second derivatives with respect to the volume or sinogram are available,
  including when trajectory or surface parameters require gradients.
- Detector and voxel spacing are scalar settings, not differentiable parameters.
- Geometry derivatives are piecewise: at exact voxel edges or corners they are one-sided.
- Distributed geometry gradients are summed by the operator. See :doc:`multi_gpu`
  for loss scaling.
