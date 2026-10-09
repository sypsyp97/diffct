Getting Started
===============

Install
-------

Projection and backprojection require an NVIDIA CUDA GPU. Geometry construction
and some validation tests can run on CPU; there is no CPU projection backend.
Install a CUDA-enabled PyTorch build appropriate for your driver first, then:

.. code-block:: bash

   python -m pip install "numpy<2.5" "numba-cuda[cu12]"
   python -m pip install diffct
   python -c "import torch, diffct; print(diffct.__version__); print(torch.cuda.is_available())"
   # from a source checkout: python examples/quickstart.py

Use ``numba-cuda[cu13]`` for a compatible CUDA 13 stack. Keep NVVM and NVJitLink
compatible with the CUDA libraries loaded by PyTorch. ``torch.cuda.is_available()``
only checks PyTorch's device access; the quickstart additionally compiles and
executes the Numba kernels. The first call includes JIT compilation overhead.
The :doc:`validation` page records previously tested environments and results;
these are not a guarantee for every CUDA/Python combination.

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
``beam="fan"`` explicitly for 2D data. Inputs must be floating-point CUDA tensors
with exactly the configured shape, without batch/channel dimensions. Computation
and outputs are float32. Fixed geometry can be supplied on CPU; the operator
stages and caches each GPU's geometry shard.

Choose a workflow
-----------------

- A custom or non-circular scan: :doc:`trajectories` covers coordinates, detector
  ordering and geometry gradients.
- An iterative reconstruction: run ``python examples/iterative_reconstruction.py
  --trajectory helical``. See :doc:`examples` for the supported methods.
- Multiple GPUs or nodes: read :doc:`distributed` before using local shards or
  differentiating losses. Extra GPUs do not pool memory for the volume.
- An existing circular-only installation: follow :doc:`migration`; the old
  scalar-angle signatures and separable-footprint backends are not drop-in APIs.

Troubleshooting
---------------

- **CUDA is unavailable:** check the selected Python environment, PyTorch CUDA
  build, GPU allocation and driver. Installing diffct does not provision a GPU.
- **Kernel compilation fails:** check Numba CUDA, NVVM and NVJitLink versions as
  a set, rather than upgrading one CUDA component independently.
- **Shape error:** use ``(D, H, W)`` for cone volumes and ``(views, U, V)`` for
  their sinograms. In distributed mode, use ``A.projection_shape`` and
  ``A.view_slice`` rather than the global view count.
- **Invalid geometry:** axes must be unit vectors with the orthogonality and
  non-degeneracy constraints described in :doc:`trajectories`.

Build these docs without a GPU
------------------------------

The documentation imports diffct for its API reference but does not execute CUDA
examples. From the repository root, with the package dependencies installed:

.. code-block:: bash

   python -m pip install sphinx sphinx-rtd-theme myst-parser
   python -m sphinx -W --keep-going -b html docs/source docs/build/html

Open ``docs/build/html/index.html`` to review this checkout's documentation.
