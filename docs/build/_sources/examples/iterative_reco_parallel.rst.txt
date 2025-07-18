Iterative Reconstruction - Parallel Beam
========================================

This example demonstrates gradient-based iterative reconstruction for parallel beam CT using PyTorch and diffct.

Overview
--------

Unlike filtered backprojection, iterative reconstruction uses optimization to minimize the difference between measured and computed projections. This example shows:

- Setting up a differentiable reconstruction pipeline
- Using gradient descent to optimize the reconstruction
- Monitoring convergence with loss curves

Code
----

.. literalinclude:: ../../../examples/iterative_reco_parallel.py
   :language: python
   :linenos:

Key Concepts
------------

1. **Iterative Approach**: Instead of direct reconstruction, we solve an optimization problem.

2. **Loss Function**: Mean squared error between measured and computed sinograms.

3. **Gradient Descent**: PyTorch's Adam optimizer updates the reconstruction iteratively.

4. **Differentiable Pipeline**: The entire reconstruction process is differentiable, enabling gradient-based optimization.

5. **Convergence Monitoring**: Loss curves show the optimization progress.

Usage
-----

.. code-block:: bash

   python examples/iterative_reco_parallel.py

The example displays the loss curve and compares the original phantom with the reconstructed image.