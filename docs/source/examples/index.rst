Examples and Tutorials
=====================

This section contains detailed examples and tutorials for using DiffCT with different CT geometries. Each example includes complete code, explanations, and visualization examples.

.. note::
   All examples assume you have DiffCT installed and a CUDA-capable GPU available. 
   See the :doc:`../quickstart` guide for installation instructions.

Available Examples
------------------

.. toctree::
   :maxdepth: 2
   :caption: Reconstruction Examples
   :name: reconstruction-examples

   parallel_beam
   fan_beam
   cone_beam

Example Overview
----------------

Choose the example that best fits your use case:

**Parallel Beam Reconstruction** (:doc:`parallel_beam`)
   Perfect for beginners and traditional CT setups. Includes both FBP and iterative reconstruction methods.
   
   * Basic forward/backward projection
   * Filtered backprojection (FBP) reconstruction
   * Iterative reconstruction with gradient descent
   * Visualization and analysis

**Fan Beam Reconstruction** (:doc:`fan_beam`)
   Ideal for clinical CT scanners and custom geometries with source-detector configurations.
   
   * Fan beam geometry setup
   * Parameter configuration and coordinate systems
   * Advanced reconstruction techniques
   * Performance considerations

**Cone Beam Reconstruction** (:doc:`cone_beam`)
   For 3D volumetric reconstruction and advanced imaging applications.
   
   * 3D cone beam geometry
   * Volumetric reconstruction
   * Memory optimization for large volumes
   * 3D visualization techniques

Getting Started
---------------

1. **New to CT reconstruction?** Start with :doc:`parallel_beam` for fundamental concepts
2. **Need specific geometry?** Jump to :doc:`fan_beam` or :doc:`cone_beam` based on your setup
3. **Looking for optimization?** All examples include gradient computation for deep learning integration

Cross-References
-----------------

* For function details: :doc:`../api/projectors`
* For geometry parameters: :doc:`../api/geometries`
* For installation help: :doc:`../quickstart`

.. seealso::
   
   The `GitHub repository <https://github.com/sypsyp97/diffct>`_ contains additional example scripts
   in the ``examples/`` directory that you can run directly.