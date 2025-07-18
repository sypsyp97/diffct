Iterative Reconstruction - Cone Beam
====================================

This example demonstrates gradient-based iterative reconstruction for 3D cone beam CT.

Overview
--------

3D cone beam iterative reconstruction is the most complex case, handling volumetric data with full 3D geometry. This example shows:

- Setting up 3D cone beam geometry
- Using 3D differentiable operators
- Optimizing volumetric reconstruction

Code
----

.. literalinclude:: ../../../examples/iterative_reco_cone.py
   :language: python
   :linenos:

Key Concepts
------------

1. **3D Volume**: Reconstruction operates on 3D arrays (D, H, W).

2. **2D Detector**: Uses detector arrays with (u, v) coordinates.

3. **Memory Requirements**: 3D operations require significant GPU memory.

4. **Computational Complexity**: Highest among all geometries due to 3D nature.

5. **Convergence**: May require careful tuning of learning rates and iterations.

Usage
-----

.. code-block:: bash

   python examples/iterative_reco_cone.py

Note: Ensure sufficient GPU memory is available for 3D operations.