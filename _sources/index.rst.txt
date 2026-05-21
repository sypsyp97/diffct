diffct: Differentiable Computed Tomography Operators
====================================================

A high-performance, CUDA-accelerated library for circular orbit CT reconstruction with end-to-end differentiable operators, enabling advanced optimization and deep learning integration in medical imaging and scientific computing.

**Key Features**
----------------
- **High Performance:** CUDA-accelerated projection and backprojection operations with optimized memory management
- **Fully Differentiable:** End-to-end gradient propagation through all CT operations for seamless deep learning integration
- **Multiple Geometries:** Support for 2D parallel-beam, 2D fan-beam, and 3D cone-beam geometries
- **PyTorch Integration:** Native PyTorch autograd support with custom CUDA kernels
- **Research Ready:** Optimized for both analytical reconstruction (FBP/FDK) and iterative methods

**Supported Geometries**
------------------------
- **Parallel Beam (2D):** Traditional parallel-beam geometry for 2D CT reconstruction
- **Fan Beam (2D):** Fan-beam geometry with configurable source-detector distances
- **Cone Beam (3D):** Full 3D cone-beam geometry for volumetric reconstruction

**Applications**
----------------
- Medical image reconstruction with deep learning enhancement
- Physics-informed neural networks for CT imaging
- Iterative reconstruction algorithms with learned priors
- Multi-modal imaging research and development
- Educational CT reconstruction demonstrations

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

Technical Report
----------------

The technical report is available as a `Preprints manuscript <https://www.preprints.org/manuscript/202605.1446/v1>`_ and by `DOI <https://doi.org/10.20944/preprints202605.1446.v1>`_.

Citation
--------

If you use this library in your research, please cite the software and the technical report:

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

   @article{202605.1446,
     doi       = {10.20944/preprints202605.1446.v1},
     url       = {https://doi.org/10.20944/preprints202605.1446.v1},
     year      = 2026,
     month     = {May},
     publisher = {Preprints},
     author    = {Yipeng Sun and Linda-Sophie Schneider and Chengze ye and Andreas Maier},
     title     = {diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories},
     journal   = {Preprints}
   }

License
-------

This project is licensed under the Apache 2.0 - see the `LICENSE <https://github.com/sun-yipeng/diffct/blob/main/LICENSE>`_ file for details.
