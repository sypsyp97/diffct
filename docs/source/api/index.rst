API Reference
=============

This section contains the complete API reference for DiffCT, including all projection operators, 
geometry configurations, and utility functions.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :name: api-documentation

   projectors
   geometries

Quick API Overview
------------------

DiffCT provides differentiable CT reconstruction operators through PyTorch autograd functions:

**Core Projection Functions** (:doc:`projectors`)
   * :class:`~diffct.differentiable.ParallelProjectorFunction` - Parallel beam forward projection
   * :class:`~diffct.differentiable.ParallelBackprojectorFunction` - Parallel beam backprojection
   * :class:`~diffct.differentiable.FanProjectorFunction` - Fan beam forward projection
   * :class:`~diffct.differentiable.FanBackprojectorFunction` - Fan beam backprojection
   * :class:`~diffct.differentiable.ConeProjectorFunction` - Cone beam forward projection
   * :class:`~diffct.differentiable.ConeBackprojectorFunction` - Cone beam backprojection

**Geometry Configuration** (:doc:`geometries`)
   * Parallel beam geometry parameters and coordinate systems
   * Fan beam geometry with source-detector configurations
   * Cone beam geometry for 3D volumetric reconstruction

Function Categories
-------------------

.. list-table:: Function Reference by Geometry
   :header-rows: 1
   :widths: 20 40 40

   * - Geometry Type
     - Forward Projection
     - Backprojection
   * - Parallel Beam
     - :class:`~diffct.differentiable.ParallelProjectorFunction`
     - :class:`~diffct.differentiable.ParallelBackprojectorFunction`
   * - Fan Beam
     - :class:`~diffct.differentiable.FanProjectorFunction`
     - :class:`~diffct.differentiable.FanBackprojectorFunction`
   * - Cone Beam
     - :class:`~diffct.differentiable.ConeProjectorFunction`
     - :class:`~diffct.differentiable.ConeBackprojectorFunction`

Usage Patterns
--------------

All projection functions follow the same pattern:

.. code-block:: python

   # Forward projection
   sinogram = ProjectorFunction.apply(image, angles, *geometry_params)
   
   # Backprojection  
   reconstruction = BackprojectorFunction.apply(sinogram, angles, *geometry_params, *image_dims)

See :doc:`../examples/index` for complete usage examples with each geometry type.

Cross-References
----------------

* **Getting Started**: :doc:`../quickstart`
* **Examples**: :doc:`../examples/index`
* **Parallel Beam Example**: :doc:`../examples/parallel_beam`
* **Fan Beam Example**: :doc:`../examples/fan_beam`
* **Cone Beam Example**: :doc:`../examples/cone_beam`

.. note::
   All functions are implemented as PyTorch autograd functions, enabling automatic 
   gradient computation for deep learning and optimization applications.