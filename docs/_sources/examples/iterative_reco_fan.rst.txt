Iterative Reconstruction - Fan Beam
===================================

This example demonstrates gradient-based iterative reconstruction for fan beam CT.

Overview
--------

Fan beam iterative reconstruction extends the parallel beam approach to handle divergent ray geometry. This example shows:

- Setting up fan beam geometry parameters
- Using differentiable fan beam operators
- Optimizing the reconstruction with gradient descent

Code
----

.. literalinclude:: ../../../examples/iterative_reco_fan.py
   :language: python
   :linenos:

Key Concepts
------------

1. **Fan Beam Geometry**: Requires source and detector distance parameters.

2. **Geometric Scaling**: The magnification factor affects the reconstruction field of view.

3. **Optimization**: Same gradient-based approach as parallel beam, but with fan beam operators.

4. **Convergence**: Typically requires more iterations due to geometric complexity.

Usage
-----

.. code-block:: bash

   python examples/iterative_reco_fan.py