Filtered Backprojection - Fan Beam
==================================

This example demonstrates 2D fan beam reconstruction using the Filtered Backprojection (FBP) algorithm with diffct.

Overview
--------

Fan beam geometry is commonly used in medical CT scanners where X-rays emanate from a point source to a linear detector array. This example shows:

- Creating a Shepp-Logan phantom
- Computing fan beam forward projection
- Applying fan beam weighting and filtering
- Performing backprojection with geometric corrections

Code
----

.. literalinclude:: ../../../examples/fbp_fan.py
   :language: python
   :linenos:

Key Concepts
------------

1. **Fan Beam Geometry**: Rays diverge from a point source to detector elements, creating geometric magnification.

2. **Weighting**: Fan beam projections must be weighted by cos(γ) where γ is the fan angle for each detector.

3. **Geometric Parameters**:
   - Source distance: Distance from X-ray source to isocenter
   - Isocenter distance: Distance from isocenter to detector

4. **Normalization**: Similar to parallel beam, but accounts for the 2π angular range.

Usage
-----

.. code-block:: bash

   python examples/fbp_fan.py

The example will display the phantom, sinogram, and reconstructed image.