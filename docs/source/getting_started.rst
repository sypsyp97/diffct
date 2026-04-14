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

Install `diffct` directly from PyPI:

.. code-block:: bash

   pip install diffct

**Verify Installation:**

.. code-block:: python

   import torch
   import diffct
   
   # Check CUDA availability
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"DiffCT version: {diffct.__version__}")

Quick Start Example
-------------------

Here's a minimal example demonstrating parallel beam projection and backprojection:

.. code-block:: python

   import torch
   import numpy as np
   from diffct import ParallelProjectorFunction, ParallelBackprojectorFunction

   # Set device
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
   # Create a simple test image (128x128)
   image = torch.zeros(128, 128, device=device)
   image[40:88, 40:88] = 1.0  # Square phantom
   
   # Define projection parameters
   num_angles = 180
   angles = torch.linspace(0, np.pi, num_angles, device=device)
   num_detectors = 128
   detector_spacing = 1.0
   
   # Forward projection
   sinogram = ParallelProjectorFunction.apply(
       image, angles, num_detectors, detector_spacing
   )
   
   # Backprojection
   reconstruction = ParallelBackprojectorFunction.apply(
       sinogram, angles, detector_spacing, 128, 128
   )
   
   print(f"Original image shape: {image.shape}")
   print(f"Sinogram shape: {sinogram.shape}")
   print(f"Reconstruction shape: {reconstruction.shape}")

Next Steps
----------

- Explore the :doc:`examples` for detailed reconstruction algorithms
- Check the :doc:`api` reference for complete function documentation
- Review the mathematical background in each example for deeper understanding
