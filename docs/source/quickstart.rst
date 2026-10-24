Quickstart
==========

This page projects an image, applies the matched adjoint and computes a gradient
with autograd. Install diffct first; see :doc:`installation`.

.. tab-set::

   .. tab-item:: CUDA

      .. code-block:: python

         import torch
         from diffct import Projector, circular_trajectory_2d_parallel

         trajectory = circular_trajectory_2d_parallel(180, device="cpu")
         A = Projector(trajectory, (64, 64), 96, beam="parallel",
                       detector_spacing=1.0, voxel_spacing=1.0)
         image = torch.zeros(A.volume_shape, device="cuda", dtype=torch.float32)
         image[20:44, 20:44] = 1.0
         measurements = A.project(image)        # (180, 96)
         adjoint = A.backproject(measurements)  # (64, 64), not an inverse

         estimate = torch.zeros_like(image, requires_grad=True)
         loss = 0.5 * (A.project(estimate) - measurements).square().sum()
         loss.backward()                       # A^T (A estimate - measurements)
         print(measurements.shape, estimate.grad.shape)

   .. tab-item:: CPU (PyTorch backend)

      .. code-block:: python

         import torch
         from diffct import Projector, circular_trajectory_2d_parallel

         trajectory = circular_trajectory_2d_parallel(180, device="cpu")
         A = Projector(trajectory, (64, 64), 96, beam="parallel",
                       detector_spacing=1.0, voxel_spacing=1.0, backend="torch")
         image = torch.zeros(A.volume_shape, dtype=torch.float32)
         image[20:44, 20:44] = 1.0
         measurements = A.project(image)        # (180, 96)
         adjoint = A.backproject(measurements)  # (64, 64), not an inverse

         estimate = torch.zeros_like(image, requires_grad=True)
         loss = 0.5 * (A.project(estimate) - measurements).square().sum()
         loss.backward()                       # A^T (A estimate - measurements)
         print(measurements.shape, estimate.grad.shape)

``Projector`` defaults to ``beam="cone"``. Set ``beam="parallel"`` or
``beam="fan"`` for 2D data. Inputs must be floating-point CPU or CUDA tensors with
the configured volume or sinogram shape in their trailing dimensions. Leading
batch/channel dimensions are preserved. Outputs use the selected precision on
the input device. Native CPU data streams through CUDA tiles; the PyTorch
backend computes on its input device. See :doc:`execution` and :doc:`chunking`.
Trajectory tensors can stay on CPU.

Next steps
----------

- A custom or non-circular scan: see :doc:`trajectories`.
- A curved detector: see :ref:`detector-surfaces` and run
  ``python examples/curved_detector.py`` from a source checkout.
- An iterative reconstruction: run ``python examples/iterative_reconstruction.py
  --trajectory helical``. See :doc:`examples`.
- Several GPUs or nodes: see :doc:`multi_gpu`.
- Code written for diffct 1.x: see :doc:`migration`.
- Gradient-based reconstruction with autograd and an optimizer: see
  :doc:`gradient_reconstruction`.

Troubleshooting
---------------

- **Shape error:** use ``(D, H, W)`` for cone volumes and ``(views, U, V)`` for
  cone sinograms. In distributed mode, use ``A.projection_shape`` and
  ``A.view_slice`` rather than the global view count.
