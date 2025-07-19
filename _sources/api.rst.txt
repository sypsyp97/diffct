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

   **Parameters:**
   
   - ``image`` (torch.Tensor): Input 2D image tensor of shape (H, W)
   - ``angles`` (torch.Tensor): Projection angles in radians, shape (num_angles,)
   - ``num_detectors`` (int): Number of detector elements
   - ``detector_spacing`` (float): Spacing between detector elements
   
   **Returns:**
   
   - ``sinogram`` (torch.Tensor): Output sinogram of shape (num_angles, num_detectors)
   
   **Example:**
   
   .. code-block:: python
   
      sinogram = ParallelProjectorFunction.apply(
          image, angles, 512, 1.0
      )

.. autoclass:: diffct.differentiable.ParallelBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

   **Parameters:**
   
   - ``sinogram`` (torch.Tensor): Input sinogram of shape (num_angles, num_detectors)
   - ``angles`` (torch.Tensor): Projection angles in radians, shape (num_angles,)
   - ``detector_spacing`` (float): Spacing between detector elements
   - ``image_width`` (int): Width of output image
   - ``image_height`` (int): Height of output image
   
   **Returns:**
   
   - ``image`` (torch.Tensor): Reconstructed image of shape (image_height, image_width)

Fan Beam Operators
-------------------

Fan beam geometry uses a point X-ray source with a fan-shaped beam, typical in medical CT scanners.

.. autoclass:: diffct.differentiable.FanProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

   **Parameters:**
   
   - ``image`` (torch.Tensor): Input 2D image tensor of shape (H, W)
   - ``angles`` (torch.Tensor): Projection angles in radians, shape (num_angles,)
   - ``num_detectors`` (int): Number of detector elements
   - ``detector_spacing`` (float): Spacing between detector elements
   - ``source_distance`` (float): Distance from rotation center to X-ray source
   - ``isocenter_distance`` (float): Distance from rotation center to detector
   
   **Returns:**
   
   - ``sinogram`` (torch.Tensor): Output sinogram of shape (num_angles, num_detectors)

.. autoclass:: diffct.differentiable.FanBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

   **Parameters:**
   
   - ``sinogram`` (torch.Tensor): Input sinogram of shape (num_angles, num_detectors)
   - ``angles`` (torch.Tensor): Projection angles in radians, shape (num_angles,)
   - ``detector_spacing`` (float): Spacing between detector elements
   - ``image_width`` (int): Width of output image
   - ``image_height`` (int): Height of output image
   - ``source_distance`` (float): Distance from rotation center to X-ray source
   - ``isocenter_distance`` (float): Distance from rotation center to detector
   
   **Returns:**
   
   - ``image`` (torch.Tensor): Reconstructed image of shape (image_height, image_width)

Cone Beam Operators
--------------------

Cone beam geometry extends fan beam to 3D with a cone-shaped X-ray beam for volumetric reconstruction.

.. autoclass:: diffct.differentiable.ConeProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

   **Parameters:**
   
   - ``volume`` (torch.Tensor): Input 3D volume tensor of shape (D, H, W)
   - ``angles`` (torch.Tensor): Projection angles in radians, shape (num_angles,)
   - ``det_u`` (int): Number of detector elements in U direction
   - ``det_v`` (int): Number of detector elements in V direction
   - ``du`` (float): Detector spacing in U direction
   - ``dv`` (float): Detector spacing in V direction
   - ``source_distance`` (float): Distance from rotation center to X-ray source
   - ``isocenter_distance`` (float): Distance from rotation center to detector
   
   **Returns:**
   
   - ``sinogram`` (torch.Tensor): Output sinogram of shape (num_angles, det_u, det_v)

.. autoclass:: diffct.differentiable.ConeBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

   **Parameters:**
   
   - ``sinogram`` (torch.Tensor): Input sinogram of shape (num_angles, det_u, det_v)
   - ``angles`` (torch.Tensor): Projection angles in radians, shape (num_angles,)
   - ``volume_width`` (int): Width of output volume
   - ``volume_height`` (int): Height of output volume  
   - ``volume_depth`` (int): Depth of output volume
   - ``du`` (float): Detector spacing in U direction
   - ``dv`` (float): Detector spacing in V direction
   - ``source_distance`` (float): Distance from rotation center to X-ray source
   - ``isocenter_distance`` (float): Distance from rotation center to detector
   
   **Returns:**
   
   - ``volume`` (torch.Tensor): Reconstructed volume of shape (volume_depth, volume_height, volume_width)

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