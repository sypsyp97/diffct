Geometry Parameters
===================

This section provides comprehensive documentation of the geometric parameters used in DiffCT's projection functions. Understanding these parameters is crucial for setting up CT reconstruction problems correctly.

Coordinate Systems and Conventions
-----------------------------------

DiffCT uses consistent coordinate systems across all geometry types to ensure predictable behavior and easy parameter translation between different geometries.

Global Coordinate System
~~~~~~~~~~~~~~~~~~~~~~~~

All geometries in DiffCT follow a right-handed coordinate system:

- **X-axis**: Horizontal, pointing right (positive direction)
- **Y-axis**: Vertical, pointing up (positive direction)  
- **Z-axis**: Depth, pointing out of the page (positive direction, 3D only)
- **Origin**: Located at the center of the reconstruction volume/image

Image and Volume Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2D Images**: Coordinate system centered at image center with extent [-cx, cx] × [-cy, cy]
  
  - cx = (Nx - 1) / 2, cy = (Ny - 1) / 2
  - Pixel (0,0) corresponds to coordinate (-cx, -cy)
  - Pixel (Nx-1, Ny-1) corresponds to coordinate (+cx, +cy)

- **3D Volumes**: Coordinate system centered at volume center with extent [-cx, cx] × [-cy, cy] × [-cz, cz]
  
  - cx = (Nx - 1) / 2, cy = (Ny - 1) / 2, cz = (Nz - 1) / 2
  - Voxel (0,0,0) corresponds to coordinate (-cx, -cy, -cz)
  - Voxel (Nx-1, Ny-1, Nz-1) corresponds to coordinate (+cx, +cy, +cz)

World Coordinates
~~~~~~~~~~~~~~~~~

- **Physical coordinates**: Measured in millimeters or other length units
- **Isocenter**: The center of rotation, typically coincident with volume center
- **Source trajectory**: Circular path around isocenter for fan/cone beam geometries
- **Detector positioning**: Fixed relative to source for fan/cone beam, independent for parallel beam

Detector Coordinates
~~~~~~~~~~~~~~~~~~~~

- **Parallel beam**: 1D linear detector array with elements indexed from 0 to num_detectors-1
- **Fan beam**: 1D linear detector array with elements indexed from 0 to num_detectors-1
- **Cone beam**: 2D detector array with (u,v) coordinates, u=horizontal, v=vertical
- **Centering convention**: Detector coordinate u=0 (or v=0) corresponds to the central detector element
- **Physical spacing**: Detector elements separated by detector_spacing (or du, dv for cone beam)

Angular Coordinates
~~~~~~~~~~~~~~~~~~~

- **Projection angles**: Measured in radians from positive X-axis
- **Rotation direction**: Counter-clockwise (right-hand rule around Z-axis)
- **Parallel beam**: Typically [0, π] radians (180°) for complete sampling
- **Fan/cone beam**: Typically [0, 2π] radians (360°) for complete sampling
- **Angular sampling**: More angles reduce streak artifacts but increase computation time

Parallel Beam Geometry
----------------------

Parallel beam geometry uses parallel rays that are perpendicular to a linear detector array. This is the simplest CT geometry and is commonly used in synchrotron CT, micro-CT, and some medical CT systems where high spatial resolution is required.

**Geometric Characteristics:**

- **Ray pattern**: All rays are parallel to each other and perpendicular to the detector array
- **Source**: Conceptually at infinite distance, creating parallel rays
- **Detector**: 1D linear array positioned perpendicular to ray direction
- **Rotation**: Detector and ray direction rotate together around the object
- **Field of view**: Determined by detector length and spacing

**Physical Interpretation:**

In parallel beam geometry, X-rays travel in parallel paths from a source at infinite distance. In practice, this is approximated by:

1. **Synchrotron sources**: Highly collimated, nearly parallel X-ray beams
2. **Collimated sources**: Using collimators to create parallel beam approximation
3. **Mathematical reconstruction**: Rebinning fan beam data into parallel beam format

**Coordinate System Details:**

- **Ray direction**: Defined by projection angle θ as (cos(θ), sin(θ))
- **Detector position**: Linear array perpendicular to ray direction
- **Ray parameterization**: Each ray defined by (angle, detector_position) pair
- **Detector coordinate**: u = (detector_index - center_index) × detector_spacing

.. code-block:: python

   # Parallel beam ray geometry for angle theta and detector position u
   ray_direction = (cos(theta), sin(theta))
   ray_start = u * (-sin(theta), cos(theta))  # Perpendicular offset from ray direction
   
   # Ray equation: point(t) = ray_start + t * ray_direction

Parameters
~~~~~~~~~~

.. list-table:: Parallel Beam Parameters
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Units
     - Description
   * - ``image``
     - torch.Tensor
     - N/A
     - 2D image tensor of shape (Nx, Ny). Must be on CUDA device for GPU acceleration.
   * - ``angles``
     - torch.Tensor
     - radians
     - Projection angles, shape (n_angles,). Typically torch.linspace(0, π, n_angles) for 180° sampling or torch.linspace(0, 2π, n_angles) for 360° sampling.
   * - ``num_detectors``
     - int
     - N/A
     - Number of detector elements in linear array. Should be ≥ √2 × max(Nx, Ny) for complete coverage.
   * - ``detector_spacing``
     - float
     - mm (or length units)
     - Physical spacing between detector elements. Determines spatial resolution of projections.

**Parameter Relationships:**

.. code-block:: python

   # Field of view calculation
   field_of_view = num_detectors * detector_spacing
   
   # Detector coordinate range
   u_min = -(num_detectors - 1) / 2 * detector_spacing
   u_max = +(num_detectors - 1) / 2 * detector_spacing
   
   # Recommended detector coverage for complete sampling
   diagonal = math.sqrt(Nx**2 + Ny**2) * pixel_size
   recommended_fov = diagonal * 1.1  # 10% margin
   
   # Angular sampling (Nyquist criterion)
   recommended_angles = math.pi * max(Nx, Ny) / 2

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

The parallel beam Radon transform is defined as:

.. math::

   p(s, \theta) = \int_{-\infty}^{\infty} f(s\cos\theta - t\sin\theta, s\sin\theta + t\cos\theta) dt

Where:
- f(x,y) is the 2D image function
- s is the detector coordinate (perpendicular distance from ray to origin)
- θ is the projection angle
- t is the parameter along the ray direction

**Discrete Implementation:**

.. code-block:: python

   # For each projection angle θ and detector position s:
   ray_direction = (cos(θ), sin(θ))
   ray_perpendicular = (-sin(θ), cos(θ))
   ray_start = s * ray_perpendicular
   
   # Integrate along ray: ray_start + t * ray_direction

Usage Guidelines and Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Angular Sampling:**
- **Minimum**: π radians (180°) for complete 2D reconstruction
- **Recommended**: 180-720 angles depending on image size and quality requirements
- **Rule of thumb**: n_angles ≥ π × max(Nx, Ny) / 2 for Nyquist sampling
- **Over-sampling**: More angles reduce streak artifacts but increase computation time

**Detector Configuration:**
- **Coverage**: num_detectors × detector_spacing should cover object diagonal
- **Resolution**: detector_spacing determines projection sampling resolution
- **Recommended**: num_detectors ≥ 1.4 × max(Nx, Ny) for complete coverage
- **Over-sampling**: Finer detector spacing improves spatial resolution

**Quality vs Performance Trade-offs:**
- **High quality**: 720 angles, fine detector spacing, large detector array
- **Balanced**: 360 angles, matched detector/pixel spacing, adequate coverage
- **Fast**: 180 angles, coarse detector spacing, minimal coverage

**Common Applications:**
- **Synchrotron CT**: High-resolution imaging with naturally parallel beams
- **Micro-CT**: When rebinned from fan beam data for artifact reduction
- **Mathematical studies**: Theoretical analysis and algorithm development
- **Iterative reconstruction**: Often preferred for regularized reconstruction methods

Fan Beam Geometry
-----------------

Fan beam geometry uses divergent rays emanating from a point X-ray source to a linear detector array. This geometry is the standard configuration for medical CT scanners and provides faster data acquisition than parallel beam while maintaining good image quality.

**Geometric Characteristics:**

- **Ray pattern**: Divergent rays emanating from a point source in a fan-shaped pattern
- **Source**: Point X-ray source rotating around the object at fixed distance
- **Detector**: 1D linear array positioned at fixed distance from source
- **Rotation**: Source and detector rotate together around the object (third-generation CT)
- **Magnification**: Objects are magnified based on source-detector geometry

**Physical Interpretation:**

Fan beam geometry models the actual physics of X-ray CT scanners:

1. **X-ray tube**: Point source emitting divergent X-ray beam
2. **Detector array**: Linear array of detector elements (scintillators + photodiodes)
3. **Gantry rotation**: Source-detector assembly rotates around patient
4. **Geometric magnification**: Objects closer to source appear larger on detector

**Source-Detector Relationship:**

The key geometric parameters define the source-detector configuration:

- **Source distance**: Distance from X-ray source to isocenter (rotation center)
- **Detector distance**: Distance from X-ray source to detector array
- **Isocenter distance**: Distance from isocenter to detector = detector_distance - source_distance
- **Magnification factor**: M = detector_distance / source_distance

.. code-block:: python

   # Fan beam geometry relationships
   total_distance = source_distance + isocenter_distance  # Source to detector
   magnification = total_distance / source_distance
   field_of_view = detector_width / magnification
   
   # Source and detector positions for angle θ
   source_x = -source_distance * sin(θ)
   source_y = source_distance * cos(θ)
   detector_center_x = isocenter_distance * sin(θ)
   detector_center_y = -isocenter_distance * cos(θ)

Parameters
~~~~~~~~~~

.. list-table:: Fan Beam Parameters
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Units
     - Description
   * - ``image``
     - torch.Tensor
     - N/A
     - 2D image tensor of shape (Nx, Ny). Must be on CUDA device for GPU acceleration.
   * - ``angles``
     - torch.Tensor
     - radians
     - Projection angles, shape (n_angles,). Typically torch.linspace(0, 2π, n_angles) for complete 360° sampling.
   * - ``num_detectors``
     - int
     - N/A
     - Number of detector elements in linear array. Should provide adequate coverage at magnified scale.
   * - ``detector_spacing``
     - float
     - mm (or length units)
     - Physical spacing between detector elements on the detector array.
   * - ``source_distance``
     - float
     - mm (or length units)
     - Distance from X-ray source to isocenter (rotation center). Should be >> object size.
   * - ``detector_distance``
     - float
     - mm (or length units)
     - **Total** distance from X-ray source to detector array. Must be > source_distance.

**Critical Parameter Relationships:**

.. code-block:: python

   # Geometric validation
   assert detector_distance > source_distance, "Detector must be farther than source"
   
   # Key derived quantities
   isocenter_distance = detector_distance - source_distance
   magnification = detector_distance / source_distance
   
   # Field of view at isocenter
   detector_width = num_detectors * detector_spacing
   field_of_view = detector_width / magnification
   
   # Fan angle (half-angle from central ray to edge)
   fan_half_angle = math.atan(detector_width / (2 * detector_distance))
   fan_full_angle = 2 * fan_half_angle
   
   # Detector coordinate for element i
   u_i = (i - (num_detectors - 1) / 2) * detector_spacing
   
   # Fan angle for detector element i
   gamma_i = math.atan(u_i / detector_distance)

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

The fan beam transform relates to the parallel beam Radon transform through geometric relationships:

.. math::

   p_{fan}(\beta, \gamma) = p_{parallel}(s, \theta)

Where the coordinate transformations are:

.. math::

   s &= D \sin(\gamma) \\
   \theta &= \beta + \gamma \\
   D &= \text{source\_distance}

**Ray Parameterization:**

For projection angle β and detector element with fan angle γ:

.. code-block:: python

   # Source position
   source_x = -source_distance * sin(β)
   source_y = source_distance * cos(β)
   
   # Detector element position
   u = detector_element * detector_spacing  # Detector coordinate
   det_x = (detector_distance - source_distance) * sin(β) + u * cos(β)
   det_y = -(detector_distance - source_distance) * cos(β) + u * sin(β)
   
   # Ray direction (normalized)
   ray_dir = (det_x - source_x, det_y - source_y) / ||(det_x - source_x, det_y - source_y)||

**Fan Beam Filtering (FBP Preprocessing):**

Fan beam projections require weighting before filtering:

.. code-block:: python

   # Cosine weighting for each detector element
   u = (detector_indices - (num_detectors - 1) / 2) * detector_spacing
   gamma = torch.atan(u / detector_distance)  # Fan angles
   weights = torch.cos(gamma)
   
   # Apply weights before ramp filtering
   weighted_projections = projections * weights

Usage Guidelines and Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Angular Sampling:**
- **Required**: 2π radians (360°) for complete sampling (unlike parallel beam)
- **Recommended**: 720-1440 angles for high-quality reconstruction
- **Minimum**: 360 angles for basic reconstruction
- **Over-sampling**: Reduces view aliasing artifacts

**Geometric Configuration:**
- **Source distance**: Should be 3-10× larger than object diameter
- **Magnification**: Typically 1.2-2.0× for good balance of resolution and field of view
- **Detector coverage**: Should encompass entire object at magnified scale
- **Fan angle**: Keep < 60° to minimize geometric distortions

**Typical Parameter Ranges:**

.. code-block:: python

   # Medical CT (human torso)
   source_distance = 1000.0      # mm
   detector_distance = 1500.0    # mm (1.5× magnification)
   num_detectors = 1024
   detector_spacing = 1.0        # mm
   
   # Micro-CT (small animals/samples)
   source_distance = 200.0       # mm
   detector_distance = 300.0     # mm (1.5× magnification)
   num_detectors = 2048
   detector_spacing = 0.1        # mm (high resolution)
   
   # Industrial CT (large objects)
   source_distance = 2000.0      # mm
   detector_distance = 2500.0    # mm (1.25× magnification)
   num_detectors = 2048
   detector_spacing = 0.5        # mm

**Quality vs Performance Trade-offs:**
- **High quality**: Large source distance, fine detector spacing, many angles
- **Balanced**: Moderate magnification, matched sampling, adequate angles
- **Fast**: Higher magnification, coarser sampling, fewer angles

**Common Applications:**
- **Medical CT**: Standard clinical imaging with optimized patient dose
- **Industrial CT**: Non-destructive testing with high spatial resolution
- **Security scanning**: Rapid imaging with moderate quality requirements
- **Micro-CT**: High-resolution imaging of small specimens

Cone Beam Geometry
------------------

Cone beam geometry extends fan beam to 3D using a 2D detector array. This enables volumetric CT reconstruction from a single circular scan, making it ideal for C-arm CT, dental CT, micro-CT, and interventional imaging applications.

**Geometric Characteristics:**

- **Ray pattern**: Cone-shaped divergent rays emanating from a point source to 2D detector
- **Source**: Point X-ray source rotating around the object in a circular trajectory
- **Detector**: 2D flat-panel detector positioned at fixed distance from source
- **Rotation**: Source and detector rotate together around the object (circular cone beam CT)
- **3D coverage**: Single circular scan provides complete 3D volume reconstruction

**Physical Interpretation:**

Cone beam geometry represents the natural 3D extension of medical CT to volumetric imaging:

1. **X-ray tube**: Point source emitting cone-shaped X-ray beam
2. **Flat-panel detector**: 2D array of detector elements (amorphous silicon, CMOS, etc.)
3. **C-arm or gantry**: Mechanical system rotating source-detector around object
4. **3D magnification**: Objects magnified in all three dimensions

**3D Source-Detector Relationship:**

The cone beam geometry is defined by the source trajectory and 2D detector positioning:

- **Source trajectory**: Circular path in xy-plane around isocenter
- **Detector orientation**: 2D array with u-axis horizontal, v-axis vertical
- **Cone angle**: 3D solid angle determined by detector size and source distance
- **3D magnification**: Uniform in all directions for circular trajectory

.. code-block:: python

   # 3D cone beam geometry relationships
   magnification = detector_distance / source_distance
   field_of_view_u = detector_u * du / magnification  # Horizontal FOV
   field_of_view_v = detector_v * dv / magnification  # Vertical FOV
   
   # Cone angles (half-angles from central ray)
   cone_angle_u = math.atan(field_of_view_u / (2 * source_distance))
   cone_angle_v = math.atan(field_of_view_v / (2 * source_distance))
   
   # 3D source and detector positions for angle θ
   source_x = -source_distance * sin(θ)
   source_y = source_distance * cos(θ)
   source_z = 0  # Source rotates in xy-plane
   
   detector_center_x = isocenter_distance * sin(θ)
   detector_center_y = -isocenter_distance * cos(θ)
   detector_center_z = 0  # Detector center at isocenter level

Parameters
~~~~~~~~~~

.. list-table:: Cone Beam Parameters
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Units
     - Description
   * - ``volume``
     - torch.Tensor
     - N/A
     - 3D volume tensor of shape (Nx, Ny, Nz). Must be on CUDA device. Memory usage: Nx×Ny×Nz×4 bytes.
   * - ``angles``
     - torch.Tensor
     - radians
     - Projection angles, shape (n_views,). Typically torch.linspace(0, 2π, n_views) for complete 360° sampling.
   * - ``detector_u``
     - int
     - N/A
     - Number of detector elements in u-direction (horizontal). Determines horizontal field of view.
   * - ``detector_v``
     - int
     - N/A
     - Number of detector elements in v-direction (vertical). Determines axial coverage.
   * - ``du``
     - float
     - mm (or length units)
     - Physical spacing between detector elements in u-direction (horizontal pixel size).
   * - ``dv``
     - float
     - mm (or length units)
     - Physical spacing between detector elements in v-direction (vertical pixel size).
   * - ``source_distance``
     - float
     - mm (or length units)
     - Distance from X-ray source to isocenter. Should be >> volume size for good approximation.
   * - ``isocenter_distance``
     - float
     - mm (or length units)
     - Distance from isocenter to detector array. **Note**: This is different from fan beam parameterization.

**Important Parameter Notes:**

.. warning::
   
   Cone beam geometry uses ``isocenter_distance`` (isocenter to detector) rather than ``detector_distance`` (source to detector) used in fan beam geometry. The relationship is:
   
   .. code-block:: python
   
      # Cone beam parameterization
      total_source_to_detector = source_distance + isocenter_distance
      
      # Equivalent fan beam parameterization would be:
      # detector_distance = source_distance + isocenter_distance

**Critical Parameter Relationships:**

.. code-block:: python

   # Geometric validation
   assert source_distance > 0, "Source distance must be positive"
   assert isocenter_distance > 0, "Isocenter distance must be positive"
   
   # Key derived quantities
   total_distance = source_distance + isocenter_distance
   magnification = total_distance / source_distance
   
   # 3D field of view at isocenter
   detector_width_u = detector_u * du
   detector_height_v = detector_v * dv
   field_of_view_u = detector_width_u / magnification
   field_of_view_v = detector_height_v / magnification
   
   # Cone angles (full angles, not half-angles)
   cone_angle_u = 2 * math.atan(detector_width_u / (2 * total_distance))
   cone_angle_v = 2 * math.atan(detector_height_v / (2 * total_distance))
   
   # Detector coordinates for element (i,j)
   u_i = (i - (detector_u - 1) / 2) * du
   v_j = (j - (detector_v - 1) / 2) * dv
   
   # 3D ray direction from source to detector element (i,j)
   # (requires trigonometric calculations for each projection angle)

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

The 3D cone beam transform is the natural extension of the 2D fan beam transform to volumetric imaging:

.. math::

   p_{cone}(\beta, u, v) = \int_0^{\infty} f(\mathbf{s} + t \cdot \mathbf{d}) dt

Where:
- **β** is the projection angle (source rotation angle)
- **(u,v)** are the 2D detector coordinates
- **s** is the 3D source position
- **d** is the 3D ray direction from source to detector pixel (u,v)
- **f(x,y,z)** is the 3D volume function

**3D Ray Parameterization:**

For projection angle β and detector pixel (u,v):

.. code-block:: python

   # 3D source position (rotates in xy-plane)
   source_x = -source_distance * sin(β)
   source_y = source_distance * cos(β)
   source_z = 0
   
   # 3D detector pixel position
   # Detector center position
   det_center_x = isocenter_distance * sin(β)
   det_center_y = -isocenter_distance * cos(β)
   det_center_z = 0
   
   # Individual pixel position (u=horizontal, v=vertical)
   pixel_x = det_center_x + u * cos(β) + v * 0  # u along detector u-axis
   pixel_y = det_center_y + u * sin(β) + v * 0  # u along detector u-axis  
   pixel_z = det_center_z + v * 1               # v along detector v-axis (vertical)
   
   # 3D ray direction (normalized)
   ray_dir_x = pixel_x - source_x
   ray_dir_y = pixel_y - source_y
   ray_dir_z = pixel_z - source_z
   ray_length = sqrt(ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2)
   ray_dir = (ray_dir_x, ray_dir_y, ray_dir_z) / ray_length

**FDK Reconstruction Preprocessing:**

Cone beam projections require 3D weighting before filtering (Feldkamp-Davis-Kress algorithm):

.. code-block:: python

   # 3D cosine weighting for each detector pixel
   u_coords = (torch.arange(detector_u) - (detector_u - 1) / 2) * du
   v_coords = (torch.arange(detector_v) - (detector_v - 1) / 2) * dv
   
   # Create 2D coordinate grids
   U, V = torch.meshgrid(u_coords, v_coords, indexing='ij')
   
   # Distance from source to each detector pixel
   total_distance = source_distance + isocenter_distance
   pixel_distances = torch.sqrt(total_distance**2 + U**2 + V**2)
   
   # FDK weighting factor
   weights = total_distance / pixel_distances
   
   # Apply weights before filtering
   weighted_projections = projections * weights

Usage Guidelines and Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Angular Sampling:**
- **Required**: 2π radians (360°) for complete 3D reconstruction
- **Recommended**: 360-720 angles depending on volume size and quality requirements
- **High quality**: 720-1440 angles for research applications
- **Cone beam artifacts**: More angles help reduce cone beam artifacts

**3D Geometric Configuration:**
- **Source distance**: Should be 5-20× larger than maximum volume dimension
- **Cone angles**: Keep both cone_angle_u and cone_angle_v < 30° to minimize artifacts
- **Detector coverage**: Must encompass entire volume in both u and v directions
- **Magnification**: Typically 1.2-3.0× for balance of resolution and field of view

**Memory and Performance Considerations:**

.. warning::
   
   3D cone beam reconstruction is the most memory and computationally intensive operation in DiffCT:
   
   - **Volume memory**: Nx×Ny×Nz×4 bytes (float32)
   - **Projection memory**: n_views×detector_u×detector_v×4 bytes
   - **GPU memory**: Ensure sufficient VRAM for both volume and projections
   - **Computation time**: Scales as O(n_views × detector_u × detector_v × Nx × Ny × Nz)

**Typical Parameter Ranges:**

.. code-block:: python

   # Dental CBCT
   source_distance = 300.0       # mm
   isocenter_distance = 200.0    # mm
   detector_u, detector_v = 1024, 1024
   du, dv = 0.2, 0.2            # mm (high resolution)
   volume_size = (256, 256, 256) # 5cm³ volume
   
   # C-arm CBCT (interventional)
   source_distance = 750.0       # mm  
   isocenter_distance = 500.0    # mm
   detector_u, detector_v = 1024, 1024
   du, dv = 0.3, 0.3            # mm
   volume_size = (512, 512, 512) # 15cm³ volume
   
   # Micro-CT (small specimens)
   source_distance = 100.0       # mm
   isocenter_distance = 50.0     # mm
   detector_u, detector_v = 2048, 2048
   du, dv = 0.05, 0.05          # mm (very high resolution)
   volume_size = (1024, 1024, 1024) # 5cm³ volume at high resolution
   
   # Industrial CT (large objects)
   source_distance = 1500.0      # mm
   isocenter_distance = 500.0    # mm
   detector_u, detector_v = 2048, 2048
   du, dv = 0.2, 0.2            # mm
   volume_size = (1024, 1024, 1024) # 20cm³ volume

**Quality vs Performance Trade-offs:**
- **High quality**: Large source distance, small cone angles, fine detector spacing, many angles
- **Balanced**: Moderate cone angles, matched voxel/detector sampling, adequate angles
- **Fast**: Higher magnification, coarser sampling, fewer angles, smaller volumes

**Cone Beam Artifacts and Mitigation:**
- **Cone beam artifacts**: Increase source distance, reduce detector size, use more angles
- **Ring artifacts**: Ensure detector calibration, use flat-field correction
- **Truncation artifacts**: Ensure adequate detector coverage in both u and v directions
- **Motion artifacts**: Minimize scan time, use motion correction algorithms

**Common Applications:**
- **Dental CT**: High-resolution imaging of teeth and jaw structures
- **C-arm CT**: Interventional imaging during medical procedures
- **Micro-CT**: Research imaging of small biological and material specimens
- **Industrial CT**: Non-destructive testing of manufactured components
- **Security scanning**: 3D imaging for threat detection

Comprehensive Parameter Selection Guide
---------------------------------------

This section provides systematic guidance for choosing optimal parameters for each geometry type based on application requirements, quality goals, and computational constraints.

General Principles
~~~~~~~~~~~~~~~~~~

**Sampling Theory Considerations:**

1. **Nyquist sampling**: Avoid aliasing by adequate sampling rates
2. **Angular sampling**: More angles reduce streak artifacts but increase computation
3. **Detector sampling**: Finer spacing improves resolution but increases memory usage
4. **Coverage requirements**: Ensure complete object coverage to avoid truncation artifacts

**Quality vs Performance Trade-offs:**

- **High quality**: Maximum sampling, large arrays, many angles → slow, high memory
- **Balanced**: Adequate sampling, moderate arrays, sufficient angles → good compromise  
- **Fast**: Minimal sampling, small arrays, few angles → fast, lower quality

Parallel Beam Parameter Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Determine Required Coverage**

.. code-block:: python

   # Calculate object diagonal
   object_diagonal = math.sqrt(object_width**2 + object_height**2)
   
   # Add 10-20% margin for safety
   required_fov = object_diagonal * 1.2
   
   # Choose detector parameters
   detector_spacing = desired_pixel_size  # Match reconstruction resolution
   num_detectors = math.ceil(required_fov / detector_spacing)
   
   # Ensure even number for symmetry
   if num_detectors % 2 == 1:
       num_detectors += 1

**Step 2: Choose Angular Sampling**

.. code-block:: python

   # Minimum angles for complete sampling
   min_angles = math.pi * max(Nx, Ny) / 2
   
   # Recommended ranges:
   # - Basic quality: min_angles to 2 × min_angles
   # - High quality: 2 × min_angles to 4 × min_angles
   # - Research quality: 4 × min_angles or more
   
   # Example for 256×256 image:
   basic_angles = 360      # π × 256 / 2 ≈ 402, rounded down for speed
   high_angles = 720       # 2 × basic for better quality
   research_angles = 1440  # 4 × basic for research applications

**Step 3: Validate Configuration**

.. code-block:: python

   # Check coverage
   fov = num_detectors * detector_spacing
   coverage_ratio = fov / object_diagonal
   assert coverage_ratio >= 1.1, f"Insufficient coverage: {coverage_ratio:.2f}"
   
   # Check angular sampling
   angular_resolution = math.pi / num_angles
   recommended_resolution = 1.0 / max(Nx, Ny)  # radians
   if angular_resolution > 2 * recommended_resolution:
       print(f"Warning: Angular undersampling detected")

Fan Beam Parameter Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Choose Source-Detector Geometry**

.. code-block:: python

   # Start with object size constraints
   object_size = max(object_width, object_height)
   
   # Source distance: 3-10× object size for good approximation
   source_distance = object_size * 5  # Conservative choice
   
   # Choose magnification based on requirements:
   # - Low magnification (1.2-1.5×): Large field of view, lower resolution
   # - Medium magnification (1.5-2.0×): Balanced approach
   # - High magnification (2.0-3.0×): High resolution, smaller field of view
   magnification = 1.5  # Balanced choice
   
   detector_distance = source_distance * magnification
   isocenter_distance = detector_distance - source_distance

**Step 2: Size Detector Array**

.. code-block:: python

   # Required detector width to cover object
   required_detector_width = object_size * magnification * 1.2  # 20% margin
   
   # Choose detector spacing
   detector_spacing = desired_resolution / magnification  # Account for magnification
   
   # Calculate number of detectors
   num_detectors = math.ceil(required_detector_width / detector_spacing)
   
   # Ensure even number and power-of-2 for FFT efficiency
   num_detectors = 2 ** math.ceil(math.log2(num_detectors))

**Step 3: Validate Geometry**

.. code-block:: python

   # Check fan angle
   detector_width = num_detectors * detector_spacing
   fan_angle = 2 * math.atan(detector_width / (2 * detector_distance))
   fan_angle_degrees = math.degrees(fan_angle)
   
   if fan_angle_degrees > 60:
       print(f"Warning: Large fan angle ({fan_angle_degrees:.1f}°) may cause artifacts")
   
   # Check field of view
   fov = detector_width / magnification
   if fov < object_size * 1.1:
       print(f"Warning: Insufficient field of view ({fov:.1f} vs {object_size:.1f})")

Cone Beam Parameter Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: 3D Geometry Configuration**

.. code-block:: python

   # Object size in all three dimensions
   object_size_xy = max(object_width, object_height)
   object_size_z = object_depth
   
   # Source distance: 5-20× largest object dimension
   source_distance = max(object_size_xy, object_size_z) * 8
   
   # Choose magnification (typically lower than fan beam due to 3D constraints)
   magnification = 1.3  # Conservative for 3D
   
   # Calculate isocenter distance (note: different from fan beam parameterization)
   total_distance = source_distance * magnification
   isocenter_distance = total_distance - source_distance

**Step 2: Size 2D Detector Array**

.. code-block:: python

   # Required detector dimensions
   required_width_u = object_size_xy * magnification * 1.2
   required_height_v = object_size_z * magnification * 1.2
   
   # Choose detector pixel sizes
   du = desired_xy_resolution / magnification
   dv = desired_z_resolution / magnification
   
   # Calculate detector array size
   detector_u = math.ceil(required_width_u / du)
   detector_v = math.ceil(required_height_v / dv)
   
   # Round to convenient sizes (powers of 2 or multiples of 64)
   detector_u = ((detector_u + 63) // 64) * 64
   detector_v = ((detector_v + 63) // 64) * 64

**Step 3: Validate 3D Geometry**

.. code-block:: python

   # Check cone angles
   detector_width = detector_u * du
   detector_height = detector_v * dv
   
   cone_angle_u = 2 * math.atan(detector_width / (2 * total_distance))
   cone_angle_v = 2 * math.atan(detector_height / (2 * total_distance))
   
   cone_angle_u_deg = math.degrees(cone_angle_u)
   cone_angle_v_deg = math.degrees(cone_angle_v)
   
   if max(cone_angle_u_deg, cone_angle_v_deg) > 30:
       print(f"Warning: Large cone angles (u:{cone_angle_u_deg:.1f}°, v:{cone_angle_v_deg:.1f}°)")
       print("Consider increasing source distance or reducing detector size")
   
   # Check memory requirements
   volume_memory = Nx * Ny * Nz * 4 / (1024**3)  # GB
   projection_memory = num_views * detector_u * detector_v * 4 / (1024**3)  # GB
   total_memory = volume_memory + projection_memory
   
   print(f"Estimated GPU memory usage: {total_memory:.2f} GB")
   if total_memory > 8:  # Typical GPU memory limit
       print("Warning: High memory usage - consider reducing volume or detector size")

Memory and Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Usage Estimation:**

.. code-block:: python

   def estimate_memory_usage(geometry_type, **params):
       """Estimate GPU memory usage for different geometries."""
       
       if geometry_type == "parallel":
           Nx, Ny = params['Nx'], params['Ny']
           n_angles, n_det = params['n_angles'], params['num_detectors']
           
           image_memory = Nx * Ny * 4 / (1024**2)  # MB
           sino_memory = n_angles * n_det * 4 / (1024**2)  # MB
           total_memory = image_memory + sino_memory
           
       elif geometry_type == "fan":
           # Similar calculation for fan beam
           pass
           
       elif geometry_type == "cone":
           Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
           n_views = params['n_views']
           det_u, det_v = params['detector_u'], params['detector_v']
           
           volume_memory = Nx * Ny * Nz * 4 / (1024**3)  # GB
           proj_memory = n_views * det_u * det_v * 4 / (1024**3)  # GB
           total_memory = volume_memory + proj_memory
       
       return total_memory

**Performance Optimization Strategies:**

1. **Batch Processing**: Process multiple angles simultaneously when memory allows
2. **Gradient Checkpointing**: Trade computation for memory in backpropagation
3. **Mixed Precision**: Use float16 where precision allows (experimental)
4. **Tiled Processing**: Process large volumes in smaller tiles
5. **Sparse Sampling**: Use fewer angles for initial iterations, increase for refinement

Application-Specific Parameter Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Medical CT (Fan Beam)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Clinical chest/abdomen CT
   angles = torch.linspace(0, 2*torch.pi, 720)  # 0.5° angular sampling
   num_detectors = 1024
   detector_spacing = 1.0  # mm
   source_distance = 1000.0  # mm
   detector_distance = 1500.0  # mm (1.5× magnification)
   
   # High-resolution cardiac CT
   angles = torch.linspace(0, 2*torch.pi, 1440)  # 0.25° angular sampling
   num_detectors = 1024
   detector_spacing = 0.5  # mm (fine resolution)
   source_distance = 1200.0  # mm
   detector_distance = 1800.0  # mm (1.5× magnification)

Dental CBCT (Cone Beam)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Standard dental CBCT
   angles = torch.linspace(0, 2*torch.pi, 360)  # 1° angular sampling
   detector_u, detector_v = 1024, 1024
   du, dv = 0.2, 0.2  # mm (high resolution)
   source_distance = 300.0  # mm
   isocenter_distance = 200.0  # mm
   volume_size = (256, 256, 256)  # 5.1cm³ FOV
   
   # High-resolution endodontic imaging
   angles = torch.linspace(0, 2*torch.pi, 720)  # 0.5° angular sampling
   detector_u, detector_v = 1024, 1024
   du, dv = 0.1, 0.1  # mm (very high resolution)
   source_distance = 250.0  # mm
   isocenter_distance = 150.0  # mm
   volume_size = (512, 512, 512)  # 5.1cm³ FOV at high resolution

Micro-CT (Cone Beam)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Biological specimen micro-CT
   angles = torch.linspace(0, 2*torch.pi, 1200)  # 0.3° angular sampling
   detector_u, detector_v = 2048, 2048
   du, dv = 0.05, 0.05  # mm (very high resolution)
   source_distance = 100.0  # mm (small scale)
   isocenter_distance = 50.0  # mm
   volume_size = (1024, 1024, 1024)  # 5.1cm³ at 50μm resolution
   
   # Materials science micro-CT
   angles = torch.linspace(0, 2*torch.pi, 1800)  # 0.2° angular sampling
   detector_u, detector_v = 4096, 4096
   du, dv = 0.025, 0.025  # mm (ultra-high resolution)
   source_distance = 50.0  # mm (very small scale)
   isocenter_distance = 25.0  # mm
   volume_size = (2048, 2048, 2048)  # 5.1cm³ at 25μm resolution

Synchrotron CT (Parallel Beam)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Standard synchrotron CT
   angles = torch.linspace(0, torch.pi, 1800)  # 0.1° angular sampling
   num_detectors = 2048
   detector_spacing = 0.65  # mm
   # No source/detector distances (parallel beam)
   
   # High-resolution synchrotron CT
   angles = torch.linspace(0, torch.pi, 3600)  # 0.05° angular sampling
   num_detectors = 4096
   detector_spacing = 0.325  # mm (2× finer sampling)
   
   # Fast synchrotron CT (time-resolved studies)
   angles = torch.linspace(0, torch.pi, 180)  # 1° angular sampling
   num_detectors = 1024
   detector_spacing = 1.3  # mm (coarser for speed)

Industrial CT (Fan/Cone Beam)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Large component inspection (fan beam)
   angles = torch.linspace(0, 2*torch.pi, 1440)  # 0.25° angular sampling
   num_detectors = 2048
   detector_spacing = 0.4  # mm
   source_distance = 2000.0  # mm (large scale)
   detector_distance = 2500.0  # mm (1.25× magnification)
   
   # Small component inspection (cone beam)
   angles = torch.linspace(0, 2*torch.pi, 720)  # 0.5° angular sampling
   detector_u, detector_v = 2048, 2048
   du, dv = 0.1, 0.1  # mm
   source_distance = 500.0  # mm
   isocenter_distance = 300.0  # mm
   volume_size = (1024, 1024, 1024)  # 10.2cm³ FOV

Security Screening (Cone Beam)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Baggage screening (fast, moderate quality)
   angles = torch.linspace(0, 2*torch.pi, 180)  # 2° angular sampling (fast)
   detector_u, detector_v = 1024, 512
   du, dv = 1.0, 1.0  # mm (coarse resolution for speed)
   source_distance = 800.0  # mm
   isocenter_distance = 400.0  # mm
   volume_size = (256, 256, 128)  # Large FOV, coarse resolution

Troubleshooting Guide
---------------------

This section provides systematic approaches to diagnosing and resolving common issues encountered when working with CT geometries in DiffCT.

Reconstruction Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~

**Streaking Artifacts**

*Symptoms*: Radial streaks emanating from high-contrast objects

*Causes and Solutions*:

.. code-block:: python

   # Insufficient angular sampling
   # Problem: n_angles too small
   current_angles = 180
   recommended_angles = math.pi * max(Nx, Ny) / 2  # Nyquist criterion
   if current_angles < recommended_angles:
       print(f"Increase angles from {current_angles} to {int(recommended_angles)}")
   
   # Inconsistent detector calibration
   # Problem: Detector response variations
   # Solution: Apply flat-field correction before reconstruction
   corrected_projections = (raw_projections - dark_field) / (flat_field - dark_field)

*Quick fixes*:
- Double the number of projection angles
- Apply additional smoothing to projections
- Use iterative reconstruction with regularization

**Ring Artifacts**

*Symptoms*: Concentric rings centered on rotation axis

*Causes and Solutions*:

.. code-block:: python

   # Detector calibration issues
   # Problem: Individual detector element response variations
   # Solution: Detector-specific calibration
   
   def remove_ring_artifacts(sinogram):
       # Simple ring artifact reduction
       sino_mean = torch.mean(sinogram, dim=0)  # Average over angles
       sino_corrected = sinogram - sino_mean.unsqueeze(0) + torch.mean(sino_mean)
       return sino_corrected
   
   # Mechanical instability
   # Problem: Detector or source position variations
   # Solution: Check mechanical alignment, use motion correction

*Quick fixes*:
- Apply sinogram-based ring artifact correction
- Verify detector flat-field calibration
- Check mechanical stability of rotation stage

**Cone Beam Artifacts (3D only)**

*Symptoms*: Shading or streaking in axial direction, especially at volume edges

*Causes and Solutions*:

.. code-block:: python

   # Cone angle too large
   detector_width = detector_u * du
   detector_height = detector_v * dv
   total_distance = source_distance + isocenter_distance
   
   cone_angle_u = 2 * math.atan(detector_width / (2 * total_distance))
   cone_angle_v = 2 * math.atan(detector_height / (2 * total_distance))
   
   max_cone_angle = max(cone_angle_u, cone_angle_v)
   if math.degrees(max_cone_angle) > 30:
       print("Cone angle too large - consider:")
       print(f"- Increase source_distance from {source_distance} to {source_distance * 1.5}")
       print(f"- Reduce detector size")
       print(f"- Use helical scanning for large volumes")

*Quick fixes*:
- Increase source distance by 50-100%
- Reduce detector array size
- Use more projection angles (720-1440)
- Apply cone beam artifact correction algorithms

**Truncation Artifacts**

*Symptoms*: Bright or dark bands at volume edges, cupping artifacts

*Causes and Solutions*:

.. code-block:: python

   # Insufficient field of view coverage
   def check_coverage(geometry_type, **params):
       if geometry_type == "parallel":
           object_diagonal = math.sqrt(Nx**2 + Ny**2) * pixel_size
           detector_fov = params['num_detectors'] * params['detector_spacing']
           coverage_ratio = detector_fov / object_diagonal
           
       elif geometry_type == "fan":
           magnification = params['detector_distance'] / params['source_distance']
           detector_fov = params['num_detectors'] * params['detector_spacing']
           object_fov = detector_fov / magnification
           object_diagonal = math.sqrt(Nx**2 + Ny**2) * pixel_size
           coverage_ratio = object_fov / object_diagonal
           
       if coverage_ratio < 1.1:
           print(f"Insufficient coverage: {coverage_ratio:.2f}")
           print("Increase detector size or reduce object size")

*Quick fixes*:
- Increase detector array size by 20-50%
- Reduce reconstruction volume size
- Apply truncation correction algorithms
- Use extrapolation techniques for missing data

Performance Issues
~~~~~~~~~~~~~~~~~~

**GPU Memory Errors**

*Symptoms*: CUDA out of memory errors, system crashes

*Diagnosis and Solutions*:

.. code-block:: python

   def diagnose_memory_usage(geometry_type, **params):
       """Diagnose and suggest memory optimizations."""
       
       if geometry_type == "cone":
           # Most memory-intensive case
           Nx, Ny, Nz = params['volume_shape']
           n_views, det_u, det_v = params['projection_shape']
           
           volume_gb = Nx * Ny * Nz * 4 / (1024**3)
           projection_gb = n_views * det_u * det_v * 4 / (1024**3)
           total_gb = volume_gb + projection_gb
           
           print(f"Volume memory: {volume_gb:.2f} GB")
           print(f"Projection memory: {projection_gb:.2f} GB")
           print(f"Total estimated: {total_gb:.2f} GB")
           
           if total_gb > 8:  # Typical GPU limit
               print("\nMemory reduction strategies:")
               print(f"- Reduce volume to {int(Nx*0.8)}³: {volume_gb*0.8**3:.2f} GB")
               print(f"- Reduce detector to {int(det_u*0.8)}²: {projection_gb*0.8**2:.2f} GB")
               print("- Use gradient checkpointing")
               print("- Process in smaller batches")

*Quick fixes*:
- Reduce volume dimensions by 20-50%
- Reduce detector array size
- Use gradient checkpointing: ``torch.utils.checkpoint.checkpoint()``
- Process projections in batches
- Use mixed precision training (experimental)

**Slow Computation**

*Symptoms*: Excessive computation time, poor GPU utilization

*Optimization Strategies*:

.. code-block:: python

   def optimize_performance(**params):
       """Suggest performance optimizations."""
       
       # Check thread block efficiency
       if 'detector_u' in params and 'detector_v' in params:
           det_u, det_v = params['detector_u'], params['detector_v']
           
           # Optimal sizes are multiples of 32 (warp size)
           if det_u % 32 != 0:
               optimal_u = ((det_u + 31) // 32) * 32
               print(f"Consider detector_u = {optimal_u} (was {det_u})")
           
           if det_v % 32 != 0:
               optimal_v = ((det_v + 31) // 32) * 32
               print(f"Consider detector_v = {optimal_v} (was {det_v})")
       
       # Check memory access patterns
       print("Performance tips:")
       print("- Use powers of 2 for array dimensions when possible")
       print("- Ensure input tensors are contiguous: tensor.contiguous()")
       print("- Use appropriate CUDA device and avoid CPU-GPU transfers")

*Quick fixes*:
- Use detector dimensions that are multiples of 32
- Ensure tensors are contiguous in memory
- Minimize CPU-GPU data transfers
- Use appropriate batch sizes for your GPU
- Enable CUDA optimizations in PyTorch

**Numerical Instability**

*Symptoms*: NaN values, extreme reconstruction values, convergence issues

*Diagnosis and Solutions*:

.. code-block:: python

   def check_numerical_stability(**params):
       """Check for numerical stability issues."""
       
       # Check parameter ranges
       if 'source_distance' in params and 'detector_distance' in params:
           src_dist = params['source_distance']
           det_dist = params['detector_distance']
           
           if det_dist <= src_dist:
               print("ERROR: detector_distance must be > source_distance")
           
           magnification = det_dist / src_dist
           if magnification > 10:
               print(f"WARNING: Very high magnification ({magnification:.1f}×)")
               print("Consider reducing detector_distance")
           
           if magnification < 1.1:
               print(f"WARNING: Very low magnification ({magnification:.1f}×)")
               print("Consider increasing detector_distance")
       
       # Check for extreme values
       print("Numerical stability checklist:")
       print("- Ensure all distances are positive and reasonable")
       print("- Check for NaN or Inf values in input data")
       print("- Verify detector spacing is appropriate for geometry")
       print("- Use float32 precision consistently")

*Quick fixes*:
- Validate all geometric parameters before reconstruction
- Check input data for NaN or infinite values
- Use reasonable parameter ranges (avoid extreme magnifications)
- Ensure consistent data types (float32)
- Add small epsilon values to avoid division by zero

Parameter Validation Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Comprehensive Validation Function**

.. code-block:: python

   def validate_geometry_parameters(geometry_type, **params):
       """Comprehensive parameter validation for all geometry types."""
       
       errors = []
       warnings = []
       
       if geometry_type == "parallel":
           # Parallel beam validation
           if params['num_detectors'] <= 0:
               errors.append("num_detectors must be positive")
           if params['detector_spacing'] <= 0:
               errors.append("detector_spacing must be positive")
               
           # Coverage check
           if 'image_size' in params:
               Nx, Ny = params['image_size']
               diagonal = math.sqrt(Nx**2 + Ny**2)
               fov = params['num_detectors'] * params['detector_spacing']
               if fov < diagonal * 1.1:
                   warnings.append(f"Detector FOV ({fov:.1f}) may be insufficient for image diagonal ({diagonal:.1f})")
       
       elif geometry_type == "fan":
           # Fan beam validation
           src_dist = params['source_distance']
           det_dist = params['detector_distance']
           
           if src_dist <= 0 or det_dist <= 0:
               errors.append("All distances must be positive")
           if det_dist <= src_dist:
               errors.append("detector_distance must be > source_distance")
               
           # Geometry checks
           magnification = det_dist / src_dist
           if magnification > 5:
               warnings.append(f"High magnification ({magnification:.1f}×) may cause artifacts")
           
           # Fan angle check
           detector_width = params['num_detectors'] * params['detector_spacing']
           fan_angle = 2 * math.atan(detector_width / (2 * det_dist))
           if math.degrees(fan_angle) > 60:
               warnings.append(f"Large fan angle ({math.degrees(fan_angle):.1f}°) may cause artifacts")
       
       elif geometry_type == "cone":
           # Cone beam validation
           src_dist = params['source_distance']
           iso_dist = params['isocenter_distance']
           
           if src_dist <= 0 or iso_dist <= 0:
               errors.append("All distances must be positive")
               
           # Memory check
           if 'volume_shape' in params and 'projection_shape' in params:
               vol_shape = params['volume_shape']
               proj_shape = params['projection_shape']
               
               vol_memory = vol_shape[0] * vol_shape[1] * vol_shape[2] * 4 / (1024**3)
               proj_memory = proj_shape[0] * proj_shape[1] * proj_shape[2] * 4 / (1024**3)
               total_memory = vol_memory + proj_memory
               
               if total_memory > 12:  # Conservative GPU memory limit
                   warnings.append(f"High memory usage ({total_memory:.1f} GB) may cause issues")
       
       # Report results
       if errors:
           print("ERRORS found:")
           for error in errors:
               print(f"  - {error}")
           return False
       
       if warnings:
           print("WARNINGS:")
           for warning in warnings:
               print(f"  - {warning}")
       
       print("Parameter validation passed!")
       return True

**Usage Example**:

.. code-block:: python

   # Validate fan beam parameters
   fan_params = {
       'source_distance': 1000.0,
       'detector_distance': 1500.0,
       'num_detectors': 1024,
       'detector_spacing': 1.0,
       'image_size': (512, 512)
   }
   
   is_valid = validate_geometry_parameters('fan', **fan_params)

Complete Parameter Reference Tables
===================================

This section provides comprehensive reference tables for all parameters used in DiffCT's projection functions, including valid ranges, typical values, and parameter selection guidance.

Parallel Beam Parameter Reference
----------------------------------

.. list-table:: Complete Parallel Beam Parameters
   :widths: 15 10 10 15 15 35
   :header-rows: 1

   * - Parameter
     - Type
     - Units
     - Valid Range
     - Typical Values
     - Description & Selection Guide
   * - ``image``
     - Tensor
     - N/A
     - (1,1) to (8192,8192)
     - (256,256) to (2048,2048)
     - 2D image tensor. Larger sizes require more memory and computation. Must be on CUDA device.
   * - ``angles``
     - Tensor
     - radians
     - [0, 2π], length ≥ 1
     - π×max(Nx,Ny)/2 to 4×π×max(Nx,Ny)/2
     - Projection angles. More angles reduce artifacts but increase computation. Use [0,π] for 180° or [0,2π] for 360°.
   * - ``num_detectors``
     - int
     - N/A
     - 1 to 8192
     - 1.4×max(Nx,Ny) to 2×max(Nx,Ny)
     - Number of detector elements. Should cover object diagonal. Powers of 2 optimize FFT performance.
   * - ``detector_spacing``
     - float
     - mm
     - 0.001 to 100
     - 0.1 to 2.0
     - Physical spacing between detectors. Smaller values give higher resolution but require more detectors for coverage.

**Parameter Relationships for Parallel Beam:**

.. list-table:: Parallel Beam Derived Quantities
   :widths: 25 25 50
   :header-rows: 1

   * - Quantity
     - Formula
     - Typical Range
   * - Field of View
     - ``num_detectors × detector_spacing``
     - 100-500 mm
   * - Angular Resolution
     - ``π / num_angles`` (for 180° scan)
     - 0.1° to 2°
   * - Spatial Resolution
     - ``detector_spacing``
     - 0.1-2.0 mm
   * - Coverage Ratio
     - ``FOV / object_diagonal``
     - 1.1-1.5 (>1.1 required)

Fan Beam Parameter Reference
----------------------------

.. list-table:: Complete Fan Beam Parameters
   :widths: 15 10 10 15 15 35
   :header-rows: 1

   * - Parameter
     - Type
     - Units
     - Valid Range
     - Typical Values
     - Description & Selection Guide
   * - ``image``
     - Tensor
     - N/A
     - (1,1) to (4096,4096)
     - (256,256) to (1024,1024)
     - 2D image tensor. Fan beam typically used for moderate-resolution applications. Must be on CUDA device.
   * - ``angles``
     - Tensor
     - radians
     - [0, 2π], length ≥ 1
     - 2π×max(Nx,Ny)/2 to 4π×max(Nx,Ny)/2
     - Projection angles. **Must** use 360° (2π) for complete sampling. More angles reduce view aliasing.
   * - ``num_detectors``
     - int
     - N/A
     - 1 to 4096
     - 512 to 2048
     - Number of detector elements. Should cover magnified object. Consider fan angle limitations.
   * - ``detector_spacing``
     - float
     - mm
     - 0.01 to 10
     - 0.2 to 2.0
     - Physical spacing on detector array. Determines projection sampling resolution.
   * - ``source_distance``
     - float
     - mm
     - 10 to 10000
     - 500 to 2000
     - Distance from source to isocenter. Larger values reduce magnification and geometric distortion.
   * - ``detector_distance``
     - float
     - mm
     - ``source_distance`` + 1 to 20000
     - 1.2×``source_distance`` to 3×``source_distance``
     - **Total** distance from source to detector. Must be > ``source_distance``. Determines magnification.

**Parameter Relationships for Fan Beam:**

.. list-table:: Fan Beam Derived Quantities
   :widths: 25 35 40
   :header-rows: 1

   * - Quantity
     - Formula
     - Typical Range
   * - Magnification
     - ``detector_distance / source_distance``
     - 1.2 to 3.0
   * - Isocenter Distance
     - ``detector_distance - source_distance``
     - 200-1500 mm
   * - Field of View
     - ``(num_detectors × detector_spacing) / magnification``
     - 200-600 mm
   * - Fan Angle
     - ``2 × atan(detector_width / (2 × detector_distance))``
     - 20° to 60°
   * - Angular Resolution
     - ``2π / num_angles``
     - 0.25° to 1°

**Fan Beam Parameter Constraints:**

.. list-table:: Fan Beam Design Constraints
   :widths: 30 70
   :header-rows: 1

   * - Constraint
     - Recommendation
   * - ``detector_distance > source_distance``
     - **Required** for valid geometry
   * - Fan angle < 60°
     - Reduces geometric artifacts
   * - Magnification 1.2-3.0
     - Balances resolution and field of view
   * - ``source_distance >> object_size``
     - Reduces local magnification variations

Cone Beam Parameter Reference
-----------------------------

.. list-table:: Complete Cone Beam Parameters
   :widths: 15 10 10 15 15 35
   :header-rows: 1

   * - Parameter
     - Type
     - Units
     - Valid Range
     - Typical Values
     - Description & Selection Guide
   * - ``volume``
     - Tensor
     - N/A
     - (1,1,1) to (2048,2048,2048)
     - (128,128,128) to (512,512,512)
     - 3D volume tensor. Large volumes require significant GPU memory (Nx×Ny×Nz×4 bytes). Must be on CUDA device.
   * - ``angles``
     - Tensor
     - radians
     - [0, 2π], length ≥ 1
     - 360 to 1440 angles
     - Projection angles. **Must** use 360° (2π) for complete 3D sampling. More angles reduce cone beam artifacts.
   * - ``detector_u``
     - int
     - N/A
     - 1 to 4096
     - 256 to 2048
     - Horizontal detector elements. Determines horizontal field of view. Should be multiple of 32 for GPU efficiency.
   * - ``detector_v``
     - int
     - N/A
     - 1 to 4096
     - 256 to 2048
     - Vertical detector elements. Determines axial coverage. Should be multiple of 32 for GPU efficiency.
   * - ``du``
     - float
     - mm
     - 0.001 to 10
     - 0.05 to 1.0
     - Horizontal detector pixel spacing. Smaller values give higher resolution but require larger arrays.
   * - ``dv``
     - float
     - mm
     - 0.001 to 10
     - 0.05 to 1.0
     - Vertical detector pixel spacing. Often equal to ``du`` for square pixels.
   * - ``source_distance``
     - float
     - mm
     - 10 to 10000
     - 100 to 2000
     - Distance from source to isocenter. Should be much larger than volume size to minimize cone beam artifacts.
   * - ``isocenter_distance``
     - float
     - mm
     - 1 to 10000
     - 50 to 1000
     - Distance from isocenter to detector. **Note**: Different from fan beam parameterization.

**Parameter Relationships for Cone Beam:**

.. list-table:: Cone Beam Derived Quantities
   :widths: 25 35 40
   :header-rows: 1

   * - Quantity
     - Formula
     - Typical Range
   * - Total Distance
     - ``source_distance + isocenter_distance``
     - 200-3000 mm
   * - Magnification
     - ``total_distance / source_distance``
     - 1.2 to 3.0
   * - Horizontal FOV
     - ``(detector_u × du) / magnification``
     - 50-300 mm
   * - Vertical FOV
     - ``(detector_v × dv) / magnification``
     - 50-300 mm
   * - Cone Angle U
     - ``2 × atan(detector_width_u / (2 × total_distance))``
     - 10° to 30°
   * - Cone Angle V
     - ``2 × atan(detector_height_v / (2 × total_distance))``
     - 10° to 30°
   * - Volume Memory
     - ``Nx × Ny × Nz × 4 bytes``
     - 64 MB to 32 GB

**Cone Beam Parameter Constraints:**

.. list-table:: Cone Beam Design Constraints
   :widths: 30 70
   :header-rows: 1

   * - Constraint
     - Recommendation
   * - ``source_distance > 0`` and ``isocenter_distance > 0``
     - **Required** for valid geometry
   * - Max cone angle < 30°
     - Critical for artifact-free reconstruction
   * - ``source_distance >> max(volume_dimensions)``
     - Reduces cone beam artifacts
   * - Total GPU memory < available VRAM
     - Monitor memory usage: volume + projections
   * - Detector dimensions multiple of 32
     - Optimizes GPU thread utilization

Memory Usage Reference
----------------------

.. list-table:: Memory Usage Estimation (Float32)
   :widths: 20 30 25 25
   :header-rows: 1

   * - Geometry
     - Configuration
     - Memory Formula
     - Example Usage
   * - Parallel
     - 512² image, 720 angles, 1024 detectors
     - ``Nx×Ny + n_angles×n_det`` × 4 bytes
     - 1 MB + 3 MB = 4 MB
   * - Fan
     - 512² image, 720 angles, 1024 detectors
     - ``Nx×Ny + n_angles×n_det`` × 4 bytes
     - 1 MB + 3 MB = 4 MB
   * - Cone
     - 256³ volume, 360 angles, 512² detector
     - ``Nx×Ny×Nz + n_views×det_u×det_v`` × 4 bytes
     - 64 MB + 360 MB = 424 MB
   * - Cone (Large)
     - 512³ volume, 720 angles, 1024² detector
     - ``Nx×Ny×Nz + n_views×det_u×det_v`` × 4 bytes
     - 512 MB + 3 GB = 3.5 GB

Performance Reference
---------------------

.. list-table:: Relative Performance Comparison
   :widths: 20 20 20 40
   :header-rows: 1

   * - Geometry
     - Computation
     - Memory
     - Scaling Factors
   * - Parallel
     - Fast
     - Low
     - O(n_angles × n_detectors × Nx × Ny)
   * - Fan
     - Fast
     - Low
     - O(n_angles × n_detectors × Nx × Ny)
   * - Cone
     - Slow
     - High
     - O(n_views × det_u × det_v × Nx × Ny × Nz)

**Performance Optimization Guidelines:**

1. **Use appropriate data types**: Float32 is typically sufficient and uses half the memory of Float64
2. **Optimize array dimensions**: Use multiples of 32 for detector dimensions when possible
3. **Balance quality vs speed**: More angles and finer sampling improve quality but increase computation time
4. **Monitor GPU memory**: Cone beam reconstructions can easily exceed GPU memory limits
5. **Consider batch processing**: Process multiple angles simultaneously when memory allows

Parameter Selection Workflow
-----------------------------

**Step-by-Step Parameter Selection:**

1. **Define Requirements**
   
   - Spatial resolution needed
   - Field of view required
   - Acceptable computation time
   - Available GPU memory

2. **Choose Geometry Type**
   
   - Parallel: Highest quality, synchrotron-like applications
   - Fan: Medical CT, balanced quality/speed
   - Cone: 3D imaging, single-scan volumetric reconstruction

3. **Size Reconstruction Volume/Image**
   
   - Based on spatial resolution and field of view requirements
   - Consider memory constraints for cone beam

4. **Configure Detector Array**
   
   - Ensure adequate coverage (>110% of object size)
   - Match detector spacing to desired resolution
   - Use efficient array sizes (powers of 2, multiples of 32)

5. **Set Geometric Parameters**
   
   - Source distance: 3-10× object size
   - Magnification: 1.2-3.0× for good balance
   - Keep cone/fan angles reasonable (<30° for cone, <60° for fan)

6. **Choose Angular Sampling**
   
   - Start with Nyquist criterion
   - Increase for better quality
   - Balance with computation time constraints

7. **Validate Configuration**
   
   - Check memory requirements
   - Verify geometric constraints
   - Test with small-scale reconstruction

This systematic approach ensures optimal parameter selection for your specific application requirements.