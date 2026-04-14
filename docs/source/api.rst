API Reference
=============

This section provides comprehensive documentation for all differentiable CT operators in `diffct`. Each function is implemented as a PyTorch autograd Function, enabling seamless gradient computation through the CT reconstruction pipeline.

Overview
--------

The `diffct` library provides six main differentiable operators organized by geometry type:

- **Parallel Beam (2D):** Traditional parallel-beam CT geometry
- **Fan Beam (2D):** Fan-beam geometry with configurable source-detector setup  
- **Cone Beam (3D):** Full 3D cone-beam geometry for volumetric reconstruction

Each geometry type includes both forward projection and backprojection operators that are fully differentiable and CUDA-accelerated.

Parallel Beam Operators
------------------------

The parallel beam geometry assumes parallel X-ray beams, commonly used in synchrotron CT and some medical CT scanners.

.. autoclass:: diffct.differentiable.ParallelProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diffct.differentiable.ParallelBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:


Fan Beam Operators
-------------------

Fan beam geometry uses a point X-ray source with a fan-shaped beam, typical in medical CT scanners.

.. autoclass:: diffct.differentiable.FanProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diffct.differentiable.FanBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:


Cone Beam Operators
--------------------

Cone beam geometry extends fan beam to 3D with a cone-shaped X-ray beam for volumetric reconstruction.

.. autoclass:: diffct.differentiable.ConeProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diffct.differentiable.ConeBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:


Usage Notes
-----------

**Memory Management:**
- All operators work with GPU tensors for optimal performance
- Ensure sufficient GPU memory for your problem size
- Use ``torch.cuda.empty_cache()`` if encountering memory issues

**Gradient Computation:**
- All operators support automatic differentiation
- Gradients flow through both forward and backward passes
- Set ``requires_grad=True`` on input tensors to enable gradients

**Performance Considerations:**
- Use contiguous tensors for optimal memory access
- Consider batch processing for multiple reconstructions
- Profile your code to identify bottlenecks

**Coordinate Systems:**
- Image/volume coordinates: (0,0) at top-left corner
- Detector coordinates: centered at detector array center
- Rotation: counter-clockwise around z-axis (right-hand rule)