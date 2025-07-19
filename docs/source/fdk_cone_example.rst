Cone Beam FDK Reconstruction
============================

This example demonstrates 3D cone beam FDK (Feldkamp-Davis-Kress) reconstruction using the `ConeProjectorFunction` and `ConeBackprojectorFunction` from `diffct`.

Overview
--------

The FDK algorithm is the standard analytical method for 3D cone beam CT reconstruction. This example shows how to:

- Configure 3D cone beam geometry with 2D detector array
- Generate cone beam projections from a 3D phantom
- Apply cosine weighting and ramp filtering for FDK reconstruction
- Perform 3D backprojection to reconstruct the volume

Mathematical Background
-----------------------

**Cone Beam Geometry**

Cone beam CT extends fan beam to 3D using a point X-ray source and 2D detector array. Key parameters:

- **Source distance** :math:`D_s`: Distance from rotation center to X-ray source
- **Detector distance** :math:`D_d`: Distance from rotation center to detector plane
- **Detector coordinates** :math:`(u, v)`: Horizontal and vertical detector positions
- **Cone angles** :math:`(\alpha, \beta)`: Horizontal and vertical beam divergence

**3D Forward Projection**

The cone beam projection at source angle :math:`\phi` and detector position :math:`(u, v)` is:

.. math::
   p(\phi, u, v) = \int_0^{\infty} f\left(\vec{r}_s(\phi) + t \cdot \vec{d}(\phi, u, v)\right) dt

where :math:`\vec{r}_s(\phi)` is the source position and :math:`\vec{d}(\phi, u, v)` is the ray direction vector.

**FDK Algorithm**

The Feldkamp-Davis-Kress algorithm performs approximate 3D reconstruction in three steps:

1. **Cosine Weighting**: Compensate for ray divergence and :math:`1/r^2` intensity falloff:

   .. math::
      p_w(\phi, u, v) = p(\phi, u, v) \cdot \frac{D_s}{\sqrt{D_s^2 + u^2 + v^2}}

2. **Row-wise Ramp Filtering**: Apply 1D ramp filter along detector rows (u-direction):

   .. math::
      p_f(\phi, u, v) = \mathcal{F}_u^{-1}\{|\omega_u| \cdot \mathcal{F}_u\{p_w(\phi, u, v)\}\}

   Each detector row is filtered independently.

3. **3D Cone Beam Backprojection**: Reconstruct volume using weighted backprojection:

   .. math::
      f(x,y,z) = \int_0^{2\pi} \frac{D_s^2}{(D_s + x\cos\phi + y\sin\phi)^2} p_f(\phi, u_{xyz}, v_{xyz}) d\phi

   where detector coordinates :math:`(u_{xyz}, v_{xyz})` for voxel :math:`(x,y,z)` are:

   .. math::
      u_{xyz} = D_s \frac{-x\sin\phi + y\cos\phi}{D_s + x\cos\phi + y\sin\phi}

   .. math::
      v_{xyz} = D_s \frac{z}{D_s + x\cos\phi + y\sin\phi}

**Implementation Steps**

1. **3D Phantom Generation**: Create 3D Shepp-Logan phantom with 10 ellipsoids
2. **Cone Beam Projection**: Generate 2D projections using `ConeProjectorFunction`
3. **Cosine Weighting**: Apply distance-dependent weights
4. **Row-wise Filtering**: Apply ramp filter to each detector row
5. **3D Backprojection**: Reconstruct volume using `ConeBackprojectorFunction`
6. **Normalization**: Scale by :math:`\frac{\pi}{N_{\text{angles}}}` factor

**3D Shepp-Logan Phantom**

The 3D phantom extends the 2D version with 10 ellipsoids representing anatomical structures:

- **Outer skull**: Large ellipsoid encompassing the head
- **Brain tissue**: Medium ellipsoids for different brain regions
- **Ventricles**: Small ellipsoids representing fluid-filled cavities
- **Lesions**: High-contrast features for reconstruction assessment

Each ellipsoid is defined by center position :math:`(x_0, y_0, z_0)`, semi-axes :math:`(a, b, c)`, rotation angles, and attenuation coefficient.

**FDK Approximations and Limitations**

The FDK algorithm makes several approximations:

- **Circular orbit**: Assumes circular source trajectory
- **Row-wise filtering**: Ramp filtering only along detector rows
- **Small cone angle**: Most accurate for limited cone angles

These approximations introduce cone beam artifacts for large cone angles, but FDK remains widely used due to computational efficiency.

.. literalinclude:: ../../examples/fdk_cone.py
   :language: python
   :linenos:
   :caption: 3D Cone Beam FDK Example
