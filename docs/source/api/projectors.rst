Projector Functions
===================

This section documents the core projection and backprojection functions in DiffCT. These functions implement differentiable CT operators for various beam geometries using CUDA-accelerated ray tracing algorithms.

Overview
--------

DiffCT provides PyTorch autograd functions for forward projection (Radon transform) and backprojection operations. Each geometry type has dedicated projector and backprojector classes that implement the forward and backward passes required for gradient-based optimization.

The projector functions use the Siddon-Joseph ray tracing algorithm for accurate volume sampling and support:

- **Parallel beam geometry**: Parallel rays for 2D CT reconstruction
- **Fan beam geometry**: Divergent rays from point source for 2D CT
- **Cone beam geometry**: 3D cone-shaped beam for volumetric CT reconstruction

Parallel Beam Projectors
------------------------

.. autoclass:: diffct.differentiable.ParallelProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: forward
   .. automethod:: backward

.. autoclass:: diffct.differentiable.ParallelBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: forward
   .. automethod:: backward

Fan Beam Projectors
-------------------

.. autoclass:: diffct.differentiable.FanProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: forward
   .. automethod:: backward

.. autoclass:: diffct.differentiable.FanBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: forward
   .. automethod:: backward

Cone Beam Projectors
--------------------

.. autoclass:: diffct.differentiable.ConeProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: forward
   .. automethod:: backward

.. autoclass:: diffct.differentiable.ConeBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: forward
   .. automethod:: backward

Usage Examples
--------------

Basic Parallel Beam Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from diffct.differentiable import ParallelProjectorFunction
   
   # Create a simple 2D image
   image = torch.randn(128, 128, device='cuda', requires_grad=True)
   
   # Define projection parameters
   angles = torch.linspace(0, torch.pi, 180, device='cuda')
   num_detectors = 128
   detector_spacing = 1.0
   
   # Forward projection
   projector = ParallelProjectorFunction.apply
   sinogram = projector(image, angles, num_detectors, detector_spacing)
   
   # Compute gradients
   loss = sinogram.sum()
   loss.backward()
   print(f"Image gradient shape: {image.grad.shape}")

Fan Beam Projection with Geometry Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from diffct.differentiable import FanProjectorFunction
   
   # Create a 2D image
   image = torch.randn(256, 256, device='cuda', requires_grad=True)
   
   # Define fan beam geometry
   angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
   num_detectors = 512
   detector_spacing = 1.0
   source_distance = 1000.0  # Distance from source to isocenter
   detector_distance = 1500.0  # Distance from source to detector
   
   # Forward projection
   projector = FanProjectorFunction.apply
   sinogram = projector(image, angles, num_detectors, detector_spacing, 
                       source_distance, detector_distance)

3D Cone Beam Projection
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from diffct.differentiable import ConeProjectorFunction
   
   # Create a 3D volume
   volume = torch.randn(128, 128, 128, device='cuda', requires_grad=True)
   
   # Define cone beam geometry
   angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
   detector_shape = (256, 256)  # (n_u, n_v)
   detector_spacing = (1.0, 1.0)  # (du, dv)
   source_distance = 1000.0
   detector_distance = 1500.0
   
   # Forward projection
   projector = ConeProjectorFunction.apply
   projections = projector(volume, angles, detector_shape[0], detector_shape[1],
                          detector_spacing[0], detector_spacing[1],
                          source_distance, detector_distance)

Performance Notes
-----------------

- All projector functions are implemented using CUDA kernels for GPU acceleration
- The Siddon-Joseph algorithm provides accurate ray-volume intersection calculations
- Bilinear (2D) and trilinear (3D) interpolation ensure smooth gradients for optimization
- Memory usage scales with volume size and number of projection angles
- For large volumes, consider using smaller batch sizes or gradient checkpointing