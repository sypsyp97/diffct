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
   * - ``parallel3d``
     - ``(ray_dir, det_center, det_u, det_v)``, each ``(N, 3)``
     - ``(D, H, W)`` / ``(N, U, V)``

Coordinates are Cartesian ``(x, y[, z])``. Volume indexing is ``[y, x]`` for 2D
and ``[z, y, x]`` for 3D. The origin is the volume centre. Use one length unit
for positions, voxel spacing and detector spacing. Voxel spacing is a single
positive scalar or a positive per-axis tuple in tensor order: ``(dy,dx)`` or
``(dz,dy,dx)``. Per-axis spacing uses the PyTorch backend; see :doc:`execution`.

Detector axes are unit direction vectors, not pixel-step vectors. The detector
spacing sets the pixel pitch. A scalar spacing applies to both 3D detector axes.
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
count. ``ray_dir`` and ``det_u`` must be orthogonal for 2D parallel beams. ``det_u``
and ``det_v`` must be orthogonal for 3D beams. ``parallel3d`` permits oblique
detectors: the unit ray direction must not be tangent to the detector plane.
Degeneracy is checked by the frame's numerical rank in float64.
Its rays are infinite lines through each pixel. It uses the PyTorch backend
on CPU or CUDA; ``backend="auto"`` selects it automatically and explicit
``backend="numba"`` is unsupported. With flat fan/cone detectors,
the source must not coincide with the detector centre or view the detector
edge-on. The native FP32 backend requires the nearer of source and detector
centre to lie within 1e6 voxel spacings of the origin. Native surface detectors validate each actual
source/pixel pair: the endpoints must remain distinct in float32, and the
nearer endpoint of every ray must lie within 1e6 voxel spacings of the origin.
Both native endpoints must lie within 1e15 voxel spacings. Portable precision
modes validate finite, nondegenerate rays in their selected dtype; see :doc:`execution`.

Arbitrary trajectories do not guarantee enough angular coverage for an inverse
problem. Use iterative methods with regularization for sparse or incomplete
acquisitions. The analytical helpers assume a circular orbit (see :doc:`api`).

.. _detector-surfaces:

Parameterized detector surfaces
-------------------------------

``Projector(..., detector_surface=surface)`` places each pixel
on a native curve. The legacy two-argument callback places pixels
on an arc, cylinder or another parameterized surface. For 3D beams it receives
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

Full-volume and streamed operations accept CPU or CUDA offsets. The operator
transfers selected offsets to the compute device before expanding world points
for that device's view shard or streamed batch. Native execution preserves dtype
promotion during composition before staging complete coordinates as float32.
The PyTorch backend uses the selected precision. See :doc:`chunking` for memory
requirements.

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
Native sampled-surface arithmetic promotes frames and offsets to one common
dtype, at least float32; float64 inputs retain float64 arithmetic. The PyTorch
backend uses the selected precision. Gradients return in each input's original
dtype. Offsets and world points are validated
on every executing batch.

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
requires gradients, and reads and validates their current values on each call.
If none require gradients, it clones the entire trajectory. For example:

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

If poses come from learnable angles or offsets, rebuild the derived tensors
inside each optimization step and pass them to the same projector:

.. code-block:: python

   from diffct import circular_trajectory_3d

   angles = torch.linspace(0, 2 * torch.pi, 91, device=volume.device)[:-1]
   angles = angles.detach().requires_grad_()
   optimizer = torch.optim.Adam([angles], lr=1e-3)
   for _ in range(10):
       optimizer.zero_grad()
       current = circular_trajectory_3d(angles, sid=80.0, sdd=128.0)
       loss = (A.project(volume, trajectory=current) - target).square().sum()
       loss.backward()
       optimizer.step()

``project``, ``backproject``, ``__call__``, ``project_into`` and
``backproject_into`` accept the keyword-only ``trajectory`` argument. It replaces
the complete tuple for that call and must keep the configured beam and view
count. Omission or ``None`` uses the constructor trajectory. Each calculation
keeps its own geometry snapshot for backward; overrides leave the default
trajectory unchanged. The ``*_into`` methods retain their numerical,
no-autograd contract. Distributed ranks must agree on whether an override is
present and supply one coherent global trajectory.

Call-time admission checks flat-detector corners and coordinate bounds, with
exact pixel checks for ambiguous views. Frames stage once per validation device.
Native curved-detector admission uses bounded batches on each compute device
after view sharding. Distributed admission errors reach all ranks before
numerical execution.

Parameterize rotations so that axes stay unit length and orthogonal. Frame
checks run at construction and on each call with live default frames or an override; sampled surface positions
are checked on every operation.

Differentiable geometry helpers
-------------------------------

``circular_trajectory_2d_parallel``, ``circular_trajectory_2d_fan``,
``circular_trajectory_3d`` and ``circular_trajectory_3d_parallel`` accept either
a view count or a nonempty floating-point angle tensor as the first argument.
Angles are in radians and may be nonuniform or unsorted. A tensor preserves its
device, dtype and autograd history. ``start_angle``, ``end_angle``, ``device``
and ``dtype`` only configure count-based generation. Parallel helpers cover
``[0, pi)`` by default; fan and cone helpers cover ``[0, 2*pi)``.
The 3D parallel helper lifts the 2D orbit into
the XY plane with ``det_v=(0,0,1)``. In both parallel helpers,
``detector_distance`` is a transverse shift along ``det_u``.

``transform_trajectory(trajectory, rotation=None, translation=None, *, beam="cone")``
applies a rigid transform about the world origin. Points become ``R @ p + t``;
directions become ``R @ d``. ``rotation`` is a proper orthogonal matrix shaped
``(d,d)`` or ``(views,d,d)``, and ``translation`` is shaped ``(d,)`` or
``(views,d)``. Use the same dtype and device as the trajectory. Both shared and
per-view transforms preserve gradients to the frames and transform parameters.
The helper validates frames and transforms; ``Projector`` checks actual
detector pixels. Surface offsets may define valid rays even when the reference
detector center coincides with the source.

.. code-block:: python

   from diffct import transform_trajectory

   shift = torch.zeros(3, device=volume.device, requires_grad=True)
   current = transform_trajectory(tuple(t.detach() for t in current), translation=shift)
   shifted = A.project(volume, trajectory=current)

Limits:

- Second derivatives with respect to geometry raise an error.
- Forward-mode derivatives (``torch.func.jvp``, ``jacfwd``) with respect to trajectory
  tensors raise an error. Use reverse mode for geometry gradients.
- Second derivatives with respect to the volume or sinogram are available,
  including when trajectory or surface parameters require gradients.
- Detector and voxel spacing are fixed physical settings, not differentiable parameters.
- Geometry derivatives are piecewise: at exact voxel edges or corners they are one-sided.
- Distributed geometry gradients are summed by the operator. See :doc:`multi_gpu`
  for loss scaling.
