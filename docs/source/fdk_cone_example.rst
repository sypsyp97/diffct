3D Cone Beam FDK Reconstruction
==============================

This example demonstrates how to use the `ConeProjectorFunction` and `ConeBackprojectorFunction` from `diffct` to perform FDK (Feldkamp-Davis-Kress) reconstruction in a 3D cone-beam geometry. The 3D Shepp-Logan phantom is used as the test object, and the reconstruction is compared to the original phantom.

Key steps:
- Generate a 3D Shepp-Logan phantom.
- Simulate cone-beam projections using the differentiable projector.
- Apply FDK weighting and a ramp filter to the sinogram.
- Reconstruct the volume using the differentiable backprojector.
- Compute the loss and gradient for demonstration.
- Visualize the phantom, sinogram, and reconstruction (mid-slice).

.. literalinclude:: ../../examples/fdk_cone.py
   :language: python
   :linenos:
   :caption: 3D Cone Beam FDK Example
