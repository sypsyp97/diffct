Parallel Beam Filtered Backprojection (FBP)
==========================================

This example demonstrates 2D parallel beam filtered backprojection (FBP) reconstruction using the `ParallelProjectorFunction` and `ParallelBackprojectorFunction` from `diffct`.

Overview
--------

Filtered backprojection is the standard analytical reconstruction method for parallel beam CT. This example shows how to:

- Generate synthetic projection data using the Shepp-Logan phantom
- Apply ramp filtering in the frequency domain
- Perform backprojection to reconstruct the image
- Visualize the reconstruction results

Mathematical Background
-----------------------

**Parallel Beam Geometry**

In parallel beam CT, X-rays are collimated into parallel beams. The projection at angle :math:`\theta` and detector position :math:`t` is given by the Radon transform:

.. math::
   p(t, \theta) = \int_{-\infty}^{\infty} f(t\cos\theta - s\sin\theta, t\sin\theta + s\cos\theta) \, ds

where :math:`f(x,y)` is the 2D attenuation coefficient distribution.

**Filtered Backprojection Algorithm**

The FBP reconstruction consists of three main steps:

1. **Forward Projection**: Compute sinogram using the Radon transform
2. **Ramp Filtering**: Apply frequency domain filter :math:`H(\omega) = |\omega|`
3. **Backprojection**: Reconstruct using filtered projections

The complete FBP formula is:

.. math::
   f(x,y) = \int_0^\pi p_f(x\cos\theta + y\sin\theta, \theta) \, d\theta

where :math:`p_f(t, \theta)` is the filtered projection:

.. math::
   p_f(t, \theta) = \mathcal{F}^{-1}\{|\omega| \cdot \mathcal{F}\{p(t, \theta)\}\}

**Implementation Steps**

1. **Phantom Generation**: Create Shepp-Logan phantom with 5 ellipses
2. **Forward Projection**: Generate sinogram using `ParallelProjectorFunction`
3. **Ramp Filtering**: Apply :math:`H(\omega) = |\omega|` filter in frequency domain
4. **Backprojection**: Reconstruct using `ParallelBackprojectorFunction`
5. **Normalization**: Scale by :math:`\frac{\pi}{N_{\text{angles}}}` factor

**Shepp-Logan Phantom**

The phantom consists of 5 ellipses representing brain tissue structures:

- **Outer skull**: Large ellipse with low attenuation
- **Brain tissue**: Medium ellipse with baseline attenuation  
- **Ventricles**: Small ellipses with fluid-like attenuation
- **Lesions**: High-contrast features for reconstruction assessment

Each ellipse is defined by center position, semi-axes, rotation angle, and attenuation coefficient.

.. literalinclude:: ../../examples/fbp_parallel.py
   :language: python
   :linenos:
   :caption: 2D Parallel Beam FBP Example
