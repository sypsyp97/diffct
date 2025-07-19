2D Parallel Beam Filtered Backprojection (FBP)
==============================================

This example demonstrates how to use the `ParallelProjectorFunction` and `ParallelBackprojectorFunction` from `diffct` to perform filtered backprojection (FBP) reconstruction in a 2D parallel-beam geometry.

Mathematical Background
-----------------------

**Parallel Beam Geometry**

In parallel beam CT, X-rays are collimated into parallel beams. The projection at angle :math:`\theta` and detector position :math:`t` is given by the Radon transform:

.. math::
   p(t, \theta) = \int_{-\infty}^{\infty} f(t\cos\theta - s\sin\theta, t\sin\theta + s\cos\theta) \, ds

where :math:`f(x,y)` is the 2D attenuation coefficient distribution.

**Filtered Backprojection Algorithm**

The FBP reconstruction formula is:

.. math::
   f(x,y) = \int_0^\pi p_f(x\cos\theta + y\sin\theta, \theta) \, d\theta

where :math:`p_f(t, \theta)` is the filtered projection:

.. math::
   p_f(t, \theta) = \int_{-\infty}^{\infty} p(t', \theta) h(t - t') \, dt'

The ramp filter :math:`h(t)` in frequency domain is:

.. math::
   H(\omega) = |\omega|

**Implementation Details**

The example implements the following steps:

1. **Forward Projection**: Compute the Radon transform using `ParallelProjectorFunction`
2. **Ramp Filtering**: Apply the ramp filter :math:`H(\omega) = |\omega|` in frequency domain
3. **Backprojection**: Reconstruct using the filtered projections with `ParallelBackprojectorFunction`
4. **Normalization**: Apply the factor :math:`\frac{\pi}{N_{\text{angles}}}` to approximate the continuous integral


**Phantom Description**

The Shepp-Logan phantom consists of 5 ellipses with different attenuation coefficients, designed to simulate brain tissue structures. Each ellipse is defined by:

- Center position :math:`(x_0, y_0)`
- Semi-axes lengths :math:`a, b`
- Rotation angle :math:`\phi`
- Attenuation value :math:`A`

.. literalinclude:: ../../examples/fbp_parallel.py
   :language: python
   :linenos:
   :caption: 2D Parallel Beam FBP Example
