2D Fan Beam Iterative Reconstruction
====================================

This example demonstrates how to use the differentiable `FanProjectorFunction` in an iterative reconstruction pipeline for 2D fan-beam geometry. The Shepp-Logan phantom is used as the test object, and a simple gradient-based optimization is performed to reconstruct the image from simulated projections.

Key steps:
- Generate a 2D Shepp-Logan phantom.
- Simulate fan-beam projections using the differentiable projector.
- Define a PyTorch-based iterative reconstruction model and pipeline.
- Optimize the reconstruction to minimize the difference between simulated and measured sinograms.
- Visualize the loss curve, original phantom, and reconstructed image.

.. literalinclude:: ../../examples/iterative_reco_fan.py
   :language: python
   :linenos:
   :caption: 2D Fan Beam Iterative Example
