3D Cone Beam Iterative Reconstruction
=====================================

This example demonstrates how to use the differentiable `ConeProjectorFunction` in an iterative reconstruction pipeline for 3D cone-beam geometry. The 3D Shepp-Logan phantom is used as the test object, and a simple gradient-based optimization is performed to reconstruct the volume from simulated projections.

Key steps:
- Generate a 3D Shepp-Logan phantom.
- Simulate cone-beam projections using the differentiable projector.
- Define a PyTorch-based iterative reconstruction model and pipeline.
- Optimize the reconstruction to minimize the difference between simulated and measured sinograms.
- Visualize the loss curve, original phantom mid-slice, and reconstructed mid-slice.

.. literalinclude:: ../../examples/iterative_reco_cone.py
   :language: python
   :linenos:
   :caption: 3D Cone Beam Iterative Example
