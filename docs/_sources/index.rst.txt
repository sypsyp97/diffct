diffct: Differentiable Computed Tomography
=========================================

**diffct** is a CUDA-based library for computed tomography (CT) projection and reconstruction with differentiable operators. It provides high-performance implementations of forward and backward projection operations for parallel, fan, and cone beam geometries, enabling end-to-end differentiable CT reconstruction pipelines.

Features
--------

* **CUDA-accelerated**: High-performance GPU implementations using Numba CUDA
* **Differentiable**: Full PyTorch autograd integration for gradient-based optimization
* **Multiple geometries**: Support for parallel, fan, and cone beam CT
* **Production-ready**: Used in medical imaging and research applications

Quick Start
-----------

.. code-block:: python

   import torch
   from diffct.differentiable import ParallelProjectorFunction

   # Create a 2D image
   image = torch.randn(128, 128, device='cuda', requires_grad=True)
   
   # Define projection parameters
   angles = torch.linspace(0, torch.pi, 180, device='cuda')
   
   # Compute forward projection
   sinogram = ParallelProjectorFunction.apply(image, angles, 128)

Installation
------------

.. code-block:: bash

   pip install diffct

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`