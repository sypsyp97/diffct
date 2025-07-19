3D Cone Beam FDK Reconstruction
===============================

This example demonstrates how to use the `ConeProjectorFunction` and `ConeBackprojectorFunction` from `diffct` to perform FDK (Feldkamp-Davis-Kress) reconstruction in a 3D cone-beam geometry.

Mathematical Background
-----------------------

**Cone Beam Geometry**

Cone beam CT extends fan beam geometry to 3D, where X-rays originate from a point source and form a cone-shaped beam illuminating a 2D detector array. The geometry is characterized by:

- Source distance :math:`D_s`: Distance from rotation center to X-ray source
- Detector distance :math:`D_d`: Distance from rotation center to detector plane
- Detector coordinates :math:`(u, v)`: Horizontal and vertical detector positions
- Cone angles :math:`(\alpha, \beta)`: Angles in horizontal and vertical planes

**3D Projection**

The cone beam projection at source angle :math:`\phi` and detector position :math:`(u, v)` is:

.. math::
   p(\phi, u, v) = \int_0^{\infty} f\left(\vec{r}_s(\phi) + t \cdot \vec{d}(\phi, u, v)\right) dt

where :math:`\vec{r}_s(\phi)` is the source position and :math:`\vec{d}(\phi, u, v)` is the ray direction vector.

**FDK Algorithm**

The Feldkamp-Davis-Kress (FDK) algorithm is an approximate reconstruction method for cone beam CT, consisting of:

1. **Cosine Weighting**: Apply distance-dependent weighting to account for ray divergence:

   .. math::
      p_w(\phi, u, v) = p(\phi, u, v) \cdot \frac{D_s}{\sqrt{D_s^2 + u^2 + v^2}}

   This weighting compensates for the :math:`1/r^2` falloff in X-ray intensity.

2. **Ramp Filtering**: Apply 1D ramp filter along detector rows (u-direction):

   .. math::
      p_f(\phi, u, v) = \mathcal{F}_u^{-1}\{|\omega_u| \cdot \mathcal{F}_u\{p_w(\phi, u, v)\}\}

   The filtering is performed independently for each detector row.

3. **3D Backprojection**: Reconstruct the volume using:

   .. math::
      f(x,y,z) = \int_0^{2\pi} \frac{D_s^2}{(D_s + x\cos\phi + y\sin\phi)^2} p_f(\phi, u_{xyz}, v_{xyz}) d\phi

   where the detector coordinates :math:`(u_{xyz}, v_{xyz})` corresponding to voxel :math:`(x,y,z)` are:

   .. math::
      u_{xyz} = D_s \frac{-x\sin\phi + y\cos\phi}{D_s + x\cos\phi + y\sin\phi}

   .. math::
      v_{xyz} = D_s \frac{z}{D_s + x\cos\phi + y\sin\phi}

**3D Shepp-Logan Phantom**

The 3D Shepp-Logan phantom extends the 2D version with 10 ellipsoids, each defined by:

- Center position :math:`(x_0, y_0, z_0)`
- Semi-axes lengths :math:`(a, b, c)`
- Rotation angles :math:`(\phi, \theta, \psi)`
- Attenuation coefficient :math:`A`

The ellipsoid equation in rotated coordinates is:

.. math::
   \frac{x'^2}{a^2} + \frac{y'^2}{b^2} + \frac{z'^2}{c^2} \leq 1


**FDK Approximations and Limitations**

The FDK algorithm makes several approximations:

1. **Circular orbit assumption**: Source follows a circular trajectory
2. **Parallel filtering**: Ramp filtering only along detector rows
3. **Cone angle approximation**: Exact only for small cone angles

These approximations introduce artifacts for large cone angles, but the algorithm remains widely used due to its computational efficiency.

**Normalization**

The final reconstruction is normalized by :math:`\frac{\pi}{N_{\text{angles}}}` to approximate the continuous integral.

.. literalinclude:: ../../examples/fdk_cone.py
   :language: python
   :linenos:
   :caption: 3D Cone Beam FDK Example
