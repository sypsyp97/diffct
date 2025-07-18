FDK Reconstruction - Cone Beam
==============================

This example demonstrates 3D cone beam reconstruction using the Feldkamp-Davis-Kress (FDK) algorithm.

Overview
--------

FDK is the standard algorithm for cone beam CT reconstruction. This example shows:

- Creating a 3D Shepp-Logan phantom
- Computing cone beam forward projection
- Applying FDK weighting and filtering
- Performing 3D backprojection

Code
----

.. literalinclude:: ../../../examples/fdk_cone.py
   :language: python
   :linenos:

Key Concepts
------------

1. **3D Phantom**: The 3D Shepp-Logan phantom extends the 2D version with ellipsoids in 3D space.

2. **Cone Beam Geometry**: Uses a point source and 2D detector array to capture volumetric data.

3. **FDK Weighting**: Projections are weighted by D/√(D² + u² + v²) where D is source distance.

4. **3D Filtering**: Ramp filter applied in the detector u-direction for each v-slice.

5. **3D Backprojection**: Distributes filtered projections back into 3D volume space.

Usage
-----

.. code-block:: bash

   python examples/fdk_cone.py

The example displays mid-slices of the phantom and reconstruction.