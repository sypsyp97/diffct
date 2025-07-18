Filtered Backprojection - Parallel Beam
=======================================

This example demonstrates how to perform 2D parallel beam reconstruction using the Filtered Backprojection (FBP) algorithm with diffct.

Overview
--------

The example shows:
- Creating a Shepp-Logan phantom
- Computing forward projection (Radon transform)
- Applying ramp filter
- Performing backprojection
- Computing reconstruction error

Code
----

.. literalinclude:: ../../../examples/fbp_parallel.py
   :language: python
   :linenos:

Key Concepts
------------

1. **Phantom Creation**: The Shepp-Logan phantom is a standard test image for CT reconstruction algorithms.

2. **Forward Projection**: The `ParallelProjectorFunction` computes the Radon transform of the image.

3. **Ramp Filter**: A frequency-domain filter that compensates for the 1/r weighting in backprojection.

4. **Backprojection**: The adjoint operation that distributes sinogram values back into the image space.

5. **Normalization**: The final reconstruction is normalized by π/num_angles to approximate the continuous integral.

Usage
-----

Run the example:

.. code-block:: bash

   python examples/fbp_parallel.py

Expected Output
---------------

The script will display three images:
- Original phantom
- Computed sinogram
- Reconstructed image

And print the reconstruction loss and gradient information.