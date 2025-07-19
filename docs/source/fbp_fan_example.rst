2D Fan Beam Filtered Backprojection (FBP)
=========================================

This example demonstrates how to use the `FanProjectorFunction` and `FanBackprojectorFunction` from `diffct` to perform filtered backprojection (FBP) reconstruction in a 2D fan-beam geometry.

Mathematical Background
-----------------------

**Fan Beam Geometry**

In fan beam CT, X-rays originate from a point source and form a fan-shaped beam. The geometry is characterized by:

- Source distance :math:`D_s`: Distance from rotation center to X-ray source
- Detector distance :math:`D_d`: Distance from rotation center to detector array
- Fan angle :math:`\gamma`: Angle between central ray and detector element

The relationship between detector position :math:`u` and fan angle :math:`\gamma` is:

.. math::
   \gamma = \arctan\left(\frac{u}{D_s}\right)

**Fan Beam Projection**

The fan beam projection at source angle :math:`\beta` and detector position :math:`u` is:

.. math::
   p(\beta, u) = \int_0^{\infty} f\left(\vec{r}_s + t \cdot \vec{d}(\beta, u)\right) dt

where :math:`\vec{r}_s` is the source position and :math:`\vec{d}(\beta, u)` is the ray direction.

**Fan Beam FBP Algorithm**

The fan beam FBP reconstruction involves three key steps:

1. **Weighting**: Apply cosine weighting to account for ray divergence:

   .. math::
      p_w(\beta, u) = p(\beta, u) \cdot \cos(\gamma)

   where :math:`\gamma = \arctan(u/D_s)`

2. **Filtering**: Apply ramp filter in frequency domain:

   .. math::
      p_f(\beta, u) = \mathcal{F}^{-1}\{|\omega| \cdot \mathcal{F}\{p_w(\beta, u)\}\}

3. **Backprojection**: Reconstruct using the fan beam backprojection formula:

   .. math::
      f(x,y) = \int_0^{2\pi} \frac{D_s^2}{(D_s + x\cos\beta + y\sin\beta)^2} p_f(\beta, u_{xy}) d\beta

   where :math:`u_{xy}` is the detector coordinate corresponding to pixel :math:`(x,y)`:

   .. math::
      u_{xy} = D_s \frac{-x\sin\beta + y\cos\beta}{D_s + x\cos\beta + y\sin\beta}

**Normalization**

The final reconstruction is normalized by :math:`\frac{\pi}{N_{\text{angles}}}` to approximate the continuous integral over the angular range [0, 2π].

**Advantages of Fan Beam Geometry**

- More realistic representation of clinical CT scanners
- Higher photon flux utilization compared to parallel beam
- Natural magnification provides higher spatial resolution
- Shorter scan times due to wider coverage per projection

.. literalinclude:: ../../examples/fbp_fan.py
   :language: python
   :linenos:
   :caption: 2D Fan Beam FBP Example
