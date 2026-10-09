Trajectories, coordinates and gradients
=======================================

Geometry is data
----------------

``Projector`` takes one row per view in each trajectory tensor. It does not select
an algorithm based on a trajectory name. The helpers create those tensors; measured
or calibrated poses can be passed directly with the same layout.

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

Coordinates are Cartesian ``(x, y[, z])``; volume indexing is ``[y, x]`` or
``[z, y, x]``. The origin is the volume centre. Use one consistent length unit
for positions, voxel spacing and detector spacing. A voxel centre along an axis
of size ``S`` is ``(i + 0.5 - S / 2) * voxel_spacing``. Voxel spacing is a single
positive isotropic scalar, not an ``(x, y, z)`` tuple.

Detector axes are unit directions, not pixel-step vectors. Cone detector shape
and spacing use ``(U, V)`` and ``(du, dv)``; a scalar spacing uses the same pitch
for both axes. Pixel ``(u, v)`` is at:

.. code-block:: text

   det_center + (u - (U - 1) / 2) * du * det_u
              + (v - (V - 1) / 2) * dv * det_v

A measured array in ``(views, V, U)`` order must be transposed before use. Confirm
the physical detector axis directions as well; reshaping alone does not correct
orientation or a handedness mismatch. Projections are line integrals; attenuation
values therefore use the reciprocal length unit of the geometry.

Custom acquisition example
--------------------------

This complete example modifies a generated helix with a per-view detector shift.
For a measured acquisition, replace the four tensors with your calibrated arrays
in the coordinate system above.

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
count. ``ray_dir`` and ``det_u`` are orthogonal for parallel beams; ``det_u`` and
``det_v`` are orthogonal for cone beams. Fan/cone sources cannot coincide with
the detector centre or view the detector edge-on. The nearer of source and
detector centre must be within 1e6 voxel spacings of the origin, and both within
1e15. These limits protect float32 ray setup; they are not a precision guarantee.

Arbitrary rays do not ensure enough angular coverage for an inverse problem.
Sparse, truncated or incomplete acquisitions can remain ill-posed. Use iterative
methods with suitable assumptions or regularization. FDK away from a circular
orbit is a heuristic baseline, not an exact general inverse.

Geometry optimization
---------------------

With fixed geometry (no component requiring gradients), ``Projector`` clones
its input geometry at construction. Mutating the original tuple later does not
update the operator: build a new one when fixed poses change.

If any component requires gradients at construction, the operator keeps references
to the tuple and reads its current values on every call. Set ``requires_grad``
first. For a direct leaf position tensor, for example:

.. code-block:: python

   # Continue from the custom acquisition above.
   target = sinogram.detach()
   learnable_center = center.clone().requires_grad_()
   B = Projector((source, learnable_center, u_axis, v_axis), (32, 32, 32),
                 (96, 64), detector_spacing=0.8)
   loss = (B.project(volume) - target).square().sum()
   loss.backward()
   print(learnable_center.grad.shape)  # (90, 3)

This uses the same target geometry, so a zero gradient is expected. For an actual
calibration loop, see ``examples/geometry_calibration.py``. If poses are derived
from learnable angles or offsets, rebuild the derived tensors and ``Projector``
inside each optimization step to get a fresh PyTorch graph.

Geometry validation happens only at construction. Parameterize rotations and
offsets so updates preserve unit axes, orthogonality and non-degeneracy. First-order
geometry derivatives follow the cell-constant model. At exact voxel edges or
corners the returned one-sided derivative can disagree with central finite
differences. Slightly perturb symmetric test geometries (for example,
``start_angle=0.1``) when checking finite differences.

Second derivatives with respect to geometry raise an error. Second derivatives
with respect to image/sinogram tensors are available with fixed geometry. Detector
and voxel spacing are scalar settings, not learnable tensor parameters. Distributed
geometry gradients are already summed by the operator; see :doc:`distributed`
for loss scaling and collective-call requirements.
