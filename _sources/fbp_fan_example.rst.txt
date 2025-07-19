Fan Beam Filtered Backprojection (FBP)
====================================

This example demonstrates 2D fan beam filtered backprojection (FBP) reconstruction using the `FanProjectorFunction` and `FanBackprojectorFunction` from `diffct`.

Overview
--------

Fan beam FBP extends parallel beam reconstruction to the more realistic fan beam geometry used in clinical CT scanners. This example shows how to:

- Configure fan beam geometry parameters (source distance, detector distance)
- Generate fan beam projections with proper weighting
- Apply ramp filtering with cosine weighting correction
- Perform fan beam backprojection reconstruction

Mathematical Background
-----------------------

**Fan Beam Geometry**

Fan beam CT uses a point X-ray source creating a fan-shaped beam. Key geometric parameters:

- **Source distance** :math:`D_s`: Distance from rotation center to X-ray source
- **Detector distance** :math:`D_d`: Distance from rotation center to detector array  
- **Fan angle** :math:`\gamma`: Angle between central ray and detector element

The detector position :math:`u` relates to fan angle :math:`\gamma` by:

.. math::
   \gamma = \arctan\left(\frac{u}{D_s}\right)

**Fan Beam Forward Projection**

The fan beam projection at source angle :math:`\beta` and detector position :math:`u` is:

.. math::
   p(\beta, u) = \int_0^{\infty} f\left(\vec{r}_s + t \cdot \vec{d}(\beta, u)\right) dt

where :math:`\vec{r}_s` is the source position and :math:`\vec{d}(\beta, u)` is the ray direction vector.

**Fan Beam FBP Algorithm**

Fan beam FBP reconstruction involves three sequential steps:

1. **Cosine Weighting**: Compensate for ray divergence:

   .. math::
      p_w(\beta, u) = p(\beta, u) \cdot \cos(\gamma) = p(\beta, u) \cdot \frac{D_s}{\sqrt{D_s^2 + u^2}}

2. **Ramp Filtering**: Apply frequency domain filter:

   .. math::
      p_f(\beta, u) = \mathcal{F}^{-1}\{|\omega| \cdot \mathcal{F}\{p_w(\beta, u)\}\}

3. **Fan Beam Backprojection**: Reconstruct using weighted backprojection:

   .. math::
      f(x,y) = \int_0^{2\pi} \frac{D_s^2}{(D_s + x\cos\beta + y\sin\beta)^2} p_f(\beta, u_{xy}) d\beta

   where the detector coordinate :math:`u_{xy}` for pixel :math:`(x,y)` is:

   .. math::
      u_{xy} = D_s \frac{-x\sin\beta + y\cos\beta}{D_s + x\cos\beta + y\sin\beta}

**Implementation Steps**

1. **Phantom Generation**: Create Shepp-Logan phantom for testing
2. **Fan Beam Projection**: Generate sinogram using `FanProjectorFunction`
3. **Cosine Weighting**: Apply divergence correction weights
4. **Ramp Filtering**: Filter each projection in frequency domain
5. **Fan Beam Backprojection**: Reconstruct using `FanBackprojectorFunction`
6. **Normalization**: Scale by :math:`\frac{\pi}{N_{\text{angles}}}` factor

**Advantages of Fan Beam Geometry**

- **Clinical Relevance**: Matches real CT scanner geometry
- **Higher Flux**: Better X-ray utilization than parallel beam
- **Natural Magnification**: Improved spatial resolution
- **Faster Acquisition**: Wider coverage per projection angle

.. literalinclude:: ../../examples/fbp_fan.py
   :language: python
   :linenos:
   :caption: 2D Fan Beam FBP Example
