Getting Started
===============

This guide will walk you through the process of setting up `diffct` and running a simple example.

Prerequisites
-------------
- A CUDA-capable GPU
- Python 3.10 or later
- The following Python libraries:
  - PyTorch
  - NumPy
  - Numba (with CUDA support)

Installation
------------

You can install `diffct` using pip:

.. code-block:: bash

   pip install diffct

Quick Example
-------------

Here is a simple example of how to use `diffct` to perform a projection and backprojection:

.. code-block:: python

   import torch
   from diffct import ParallelProjectorFunction, ParallelBackprojectorFunction

   # Create a sample image
   image = torch.randn(1, 1, 128, 128, device='cuda')

   # Define the projector and backprojector
   projector = ParallelProjectorFunction.apply
   backprojector = ParallelBackprojectorFunction.apply

   # Perform the projection
   sinogram = projector(image)

   # Perform the backprojection
   reconstruction = backprojector(sinogram)

   print("Reconstruction shape:", reconstruction.shape)
