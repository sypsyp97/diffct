Getting Started
===============

Install
-------

diffct needs an NVIDIA CUDA GPU for projection and backprojection. Geometry
construction runs on CPU. Install a CUDA-enabled PyTorch build that matches your
driver first, then:

.. code-block:: bash

   python -m pip install "diffct[cu12]"
   python -c "import torch, diffct; print(diffct.__version__); print(torch.cuda.is_available())"
   # from a source checkout: python examples/quickstart.py

Use ``diffct[cu13]`` for CUDA 13. The extra installs the CUDA compiler libraries
that Numba CUDA needs. A plain ``pip install diffct`` does not install them; use it
only if a CUDA Toolkit is already installed on the system. The NVVM and NVJitLink packages must match the
CUDA libraries that PyTorch loads.

Minimal forward / adjoint / gradient example
--------------------------------------------

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

``Projector`` defaults to ``beam="cone"``. Set ``beam="parallel"`` or
``beam="fan"`` for 2D data. Inputs must be floating-point CPU or CUDA tensors with
exactly the configured volume or sinogram shape, without batch or channel
dimensions. Outputs are float32 on the input device. CPU data automatically
streams through CUDA tiles; see :doc:`chunking`. Trajectory tensors can stay
on CPU.

Choose a workflow
-----------------

- A custom or non-circular scan: see :doc:`trajectories`.
- A curved detector: see :ref:`detector-surfaces` and run
  ``python examples/curved_detector.py`` from a source checkout.
- An iterative reconstruction: run ``python examples/iterative_reconstruction.py
  --trajectory helical``. See :doc:`examples`.
- Several GPUs or nodes: see :doc:`multi_gpu`.
- Code written for diffct 1.x: see :doc:`migration`.

Troubleshooting
---------------

- **CUDA is unavailable:** check that the active Python environment has a CUDA
  build of PyTorch and that the GPU is visible to the process.
- **Kernel compilation fails:** use Numba CUDA, NVVM and NVJitLink versions that
  match each other. Do not upgrade only one of them.
- **Shape error:** use ``(D, H, W)`` for cone volumes and ``(views, U, V)`` for
  cone sinograms. In distributed mode, use ``A.projection_shape`` and
  ``A.view_slice`` rather than the global view count.
