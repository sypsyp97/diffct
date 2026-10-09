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
Pixel ``(u, v)`` is at:

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
and ``det_v`` must be orthogonal for cone beams. For fan and cone beams, the
source must not coincide with the detector centre or view the detector edge-on. The nearer of source and
detector centre must lie within 1e6 voxel spacings of the origin.

Arbitrary trajectories do not guarantee enough angular coverage for an inverse
problem. Use iterative methods with regularization for sparse or incomplete
acquisitions. The analytical helpers assume a circular orbit (see :doc:`api`).

Geometry gradients
------------------

Gradients with respect to trajectory tensors are first order only. Set
``requires_grad=True`` on each tensor before you construct ``Projector``. The
projector keeps references to the tensors and reads their current values on each
call. For example:

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
axes stay unit length and orthogonal. Geometry checks run only at construction.

Limits:

- Second derivatives with respect to geometry raise an error.
- Second derivatives with respect to the volume or sinogram are available with
  fixed geometry.
- Detector and voxel spacing are scalar settings, not differentiable parameters.
- Geometry derivatives are piecewise: at exact voxel edges or corners they are one-sided.
- Distributed geometry gradients are summed by the operator. See :doc:`multi_gpu`
  for loss scaling.
