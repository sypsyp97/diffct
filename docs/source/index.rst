DiffCT: Differentiable Computed Tomography Operators
====================================================

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square
   :target: https://doi.org/10.5281/zenodo.14999333
   :alt: DOI

.. image:: https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white
   :target: https://pypi.org/project/diffct/
   :alt: PyPI version

A high-performance, CUDA-accelerated library for circular orbits CT reconstruction with end-to-end differentiable operators, enabling advanced optimization and deep learning integration.

⭐ **Please star this project if you find it useful!**

Features
--------

✨ **Fast**: CUDA-accelerated projection and backprojection operations for high-performance computing

🔄 **Differentiable**: End-to-end gradient propagation enabling seamless integration with PyTorch and deep learning workflows

📐 **Multiple Geometries**: Support for parallel beam, fan beam, and cone beam CT geometries

🧠 **Deep Learning Ready**: Built for modern AI/ML pipelines with automatic differentiation support

Supported Geometries
--------------------

* **Parallel Beam**: 2D parallel-beam geometry for traditional CT reconstruction
* **Fan Beam**: 2D fan-beam geometry with configurable source-detector distances  
* **Cone Beam**: 3D cone-beam geometry for volumetric reconstruction

Quick Installation
------------------

.. code-block:: bash

   # Install from PyPI
   pip install diffct

   # Or install from source
   git clone https://github.com/sypsyp97/diffct.git
   cd diffct
   pip install -r requirements.txt
   pip install .

Prerequisites
~~~~~~~~~~~~~

* CUDA-capable GPU
* Python 3.10+
* PyTorch with CUDA support
* NumPy, Numba with CUDA support

Basic Usage Example
-------------------

Here's a simple example showing parallel beam projection and backprojection:

.. code-block:: python

   import torch
   import numpy as np
   from diffct import ParallelProjectorFunction, ParallelBackprojectorFunction

   # Create a simple phantom
   phantom = torch.randn(256, 256, device='cuda', requires_grad=True)
   
   # Define projection parameters
   angles = torch.linspace(0, 2*np.pi, 180, device='cuda')
   num_detectors = 512
   detector_spacing = 1.0
   
   # Forward projection
   sinogram = ParallelProjectorFunction.apply(
       phantom, angles, num_detectors, detector_spacing
   )
   
   # Backprojection
   reconstruction = ParallelBackprojectorFunction.apply(
       sinogram, angles, detector_spacing, 256, 256
   )
   
   # Compute loss and gradients
   loss = torch.mean((reconstruction - phantom)**2)
   loss.backward()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: getting-started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :name: api-reference
   
   api/index
   api/projectors
   api/geometries

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   :name: examples-tutorials
   
   examples/index
   examples/parallel_beam
   examples/fan_beam
   examples/cone_beam

Navigation Guide
----------------

New to DiffCT? Start here:

1. :doc:`quickstart` - Get up and running quickly with installation and basic examples
2. :doc:`api/projectors` - Learn about the core projection and backprojection functions
3. :doc:`examples/parallel_beam` - Try your first reconstruction example

For specific use cases:

* **Medical Imaging**: See :doc:`examples/parallel_beam` and :doc:`examples/cone_beam`
* **Industrial NDT**: Check :doc:`examples/fan_beam` for custom geometries
* **Deep Learning Integration**: All examples show gradient computation
* **Performance Optimization**: See :doc:`api/geometries` for parameter tuning

Quick Reference Links
---------------------

* :ref:`genindex` - Complete index of all functions and classes
* :ref:`modindex` - Module index for easy navigation
* :ref:`search` - Search the entire documentation

Key Applications
----------------

* **Medical Imaging**: CT reconstruction with advanced regularization
* **Industrial NDT**: Non-destructive testing with custom geometries
* **Deep Learning**: Training neural networks for CT reconstruction
* **Inverse Problems**: Gradient-based optimization for image reconstruction
* **Research**: Prototyping new CT reconstruction algorithms

Citation
--------

If you use this library in your research, please cite:

.. code-block:: bibtex

   @software{DiffCT2025,
     author       = {Yipeng Sun},
     title        = {DiffCT: Differentiable Computed Tomography 
                    Reconstruction with CUDA},
     year         = 2025,
     publisher    = {Zenodo},
     doi          = {10.5281/zenodo.14999333},
     url          = {https://doi.org/10.5281/zenodo.14999333}
   }

License
-------

This project is licensed under the Apache 2.0 License - see the `LICENSE <https://github.com/sypsyp97/diffct/blob/main/LICENSE>`_ file for details.

Acknowledgements
----------------

This project was highly inspired by:

* `PYRO-NN <https://github.com/csyben/PYRO-NN>`_
* `geometry_gradients_CT <https://github.com/mareikethies/geometry_gradients_CT>`_

Issues and contributions are welcome on `GitHub <https://github.com/sypsyp97/diffct>`_!

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`