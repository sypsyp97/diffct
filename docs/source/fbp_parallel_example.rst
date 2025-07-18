2D Parallel Beam Filtered Backprojection (FBP)
==============================================

This example demonstrates how to use the `ParallelProjectorFunction` and `ParallelBackprojectorFunction` from `diffct` to perform filtered backprojection (FBP) reconstruction in a 2D parallel-beam geometry. The Shepp-Logan phantom is used as the test object, and the reconstruction is compared to the original phantom.

Key steps:
- Generate a 2D Shepp-Logan phantom.
- Simulate parallel-beam projections using the differentiable projector.
- Apply a ramp filter to the sinogram.
- Reconstruct the image using the differentiable backprojector.
- Compute the loss and gradient for demonstration.
- Visualize the phantom, sinogram, and reconstruction.

.. literalinclude:: ../../examples/fbp_parallel.py
   :language: python
   :linenos:
   :caption: 2D Parallel Beam FBP Example
