diffct: Differentiable Computed Tomography Operators
====================================================

A high-performance, CUDA-accelerated library for circular orbits CT reconstruction with end-to-end differentiable operators, enabling advanced optimization and deep learning integration.

**Features**
------------
- **Fast:** CUDA-accelerated projection and backprojection operations
- **Differentiable:** End-to-end gradient propagation for deep learning workflows

**Supported Geometries**
------------------------
- **Parallel Beam:** 2D parallel-beam geometry
- **Fan Beam:** 2D fan-beam geometry
- **Cone Beam:** 3D cone-beam geometry

Getting Started
---------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   api
   examples

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

This project is licensed under the Apache 2.0 - see the `LICENSE <https://github.com/sun-yipeng/diffct/blob/main/LICENSE>`_ file for details.