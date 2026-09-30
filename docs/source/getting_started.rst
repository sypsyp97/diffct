Getting Started
===============

This guide will walk you through the process of setting up `diffct` and running your first CT reconstruction example.

Prerequisites
-------------

**Hardware Requirements:**
- CUDA-capable GPU (compute capability 6.0 or higher recommended)
- Minimum 4GB GPU memory for basic examples

**Software Requirements:**
- Python 3.10 or later
- CUDA Toolkit 11.0 or later
- Required Python packages:
  - PyTorch (with CUDA support)
  - NumPy
  - Numba (with CUDA support)

Installation
------------

Install this local arbitrary-trajectory candidate from its checkout. The
currently published PyPI release does not contain this candidate's high-level API:

.. code-block:: bash

   pip install -e .

**Verify Installation:**

.. code-block:: python

   import torch
   import diffct
   
   # Check CUDA availability
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"DiffCT version: {diffct.__version__}")

Quick Start Example
-------------------

Here's a minimal example that uses the new geometry helpers and projector API:

.. code-block:: python

   import torch
   from diffct import Projector
   from diffct.geometry import circular_trajectory_2d_parallel

   # Set device
   device = torch.device('cuda')

   # Create a simple test image (128x128)
   image = torch.zeros((128, 128), device=device)
   image[40:88, 40:88] = 1.0  # Square phantom

   # Define projection parameters
   num_views = 180
   num_detectors = 128
   detector_spacing = 1.0
   voxel_spacing = 1.0

   # Generate parallel-beam geometry
   trajectory = circular_trajectory_2d_parallel(num_views, device='cpu')
   operator = Projector(trajectory, image.shape, num_detectors,
                        beam='parallel', detector_spacing=detector_spacing,
                        voxel_spacing=voxel_spacing)

   # Forward projection
   sinogram = operator.project(image)

   # Backprojection
   reconstruction = operator.backproject(sinogram)

   print(f"Original image shape: {image.shape}")
   print(f"Sinogram shape: {sinogram.shape}")
   print(f"Reconstruction shape: {reconstruction.shape}")

``backproject`` is the matched adjoint, not an inverse reconstruction. Supply
any valid per-view trajectory tuple to the same interface. For cone beam, use a
``(depth, height, width)`` volume and a ``(detector_u, detector_v)`` detector.
Add ``devices=['cuda:0', 'cuda:1']`` to share views between local GPUs, or initialize
a process group and use ``distributed=True`` for view sharding across ranks.
See ``docs/DISTRIBUTED.md`` for gradient conventions and launch instructions.

Next Steps
----------

- Explore the :doc:`examples` for detailed reconstruction algorithms
- Check the :doc:`api` reference for complete function documentation
- Review the mathematical background in each example for deeper understanding
