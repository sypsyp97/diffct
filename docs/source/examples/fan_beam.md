# Fan Beam Examples

This section demonstrates how to use DiffCT for fan beam CT reconstruction using both Filtered Back Projection (FBP) and iterative reconstruction methods. Fan beam geometry is more realistic than parallel beam and is commonly used in medical CT scanners.

## Overview

Fan beam geometry uses a point X-ray source that creates a fan-shaped beam through the object. This geometry is more complex than parallel beam but provides better dose efficiency and is the standard in clinical CT scanners. DiffCT provides differentiable operators for both forward projection and backprojection in fan beam geometry.

## Fan Beam Geometry Fundamentals

### Coordinate System

In fan beam geometry, the key geometric parameters are:

- **Source Distance (R)**: Distance from X-ray source to rotation center
- **Isocenter Distance (D)**: Distance from rotation center to detector array  
- **Detector Spacing**: Physical spacing between detector elements
- **Fan Angle (γ)**: Angle from central ray to each detector element

### Geometric Relationships

```
Total Source-to-Detector Distance = R + D
Fan Angle γ = arctan(u / R)
where u = detector position relative to central detector
```

The fan beam geometry creates natural magnification and requires special handling for reconstruction.

## Example 1: Fan Beam Filtered Back Projection (FBP)

Fan beam FBP requires additional weighting and geometric corrections compared to parallel beam FBP.

### Complete Code Example

```python
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffct.differentiable import FanProjectorFunction, FanBackprojectorFunction

def shepp_logan_2d(Nx, Ny):
    """Generate 2D Shepp-Logan phantom for testing."""
    Nx = int(Nx)
    Ny = int(Ny)
    phantom = np.zeros((Nx, Ny), dtype=np.float32)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 1.0),
        (0.0, -0.0184, 0.6624, 0.8740, 0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18.0, -0.8),
        (-0.22, 0.0, 0.16, 0.41, 18.0, -0.8),
        (0.0, 0.35, 0.21, 0.25, 0, 0.7),
    ]
    cx = (Nx - 1) * 0.5
    cy = (Ny - 1) * 0.5
    for ix in range(Nx):
        for iy in range(Ny):
            xnorm = (ix - cx) / (Nx / 2)
            ynorm = (iy - cy) / (Ny / 2)
            val = 0.0
            for (x0, y0, a, b, angdeg, ampl) in ellipses:
                th = np.deg2rad(angdeg)
                xprime = (xnorm - x0) * np.cos(th) + (ynorm - y0) * np.sin(th)
                yprime = -(xnorm - x0) * np.sin(th) + (ynorm - y0) * np.cos(th)
                if xprime * xprime / (a * a) + yprime * yprime / (b * b) <= 1.0:
                    val += ampl
            phantom[ix, iy] = val
    phantom = np.clip(phantom, 0.0, 1.0)
    return phantom

def ramp_filter(sinogram_tensor):
    """Apply ramp filter for FBP reconstruction."""
    device = sinogram_tensor.device
    num_views, num_det = sinogram_tensor.shape
    freqs = torch.fft.fftfreq(num_det, device=device)
    omega = 2.0 * torch.pi * freqs
    ramp = torch.abs(omega)
    ramp_2d = ramp.reshape(1, num_det)
    sino_fft = torch.fft.fft(sinogram_tensor, dim=1)
    filtered_fft = sino_fft * ramp_2d
    filtered = torch.real(torch.fft.ifft(filtered_fft, dim=1))
    return filtered

# Setup fan beam geometry parameters
Nx, Ny = 256, 256
phantom = shepp_logan_2d(Nx, Ny)
num_angles = 360
angles_np = np.linspace(0, 2*math.pi, num_angles, endpoint=False).astype(np.float32)

# Fan beam specific parameters
num_detectors = 600
detector_spacing = 1.0
source_distance = 800.0      # Distance from source to rotation center
isocenter_distance = 500.0   # Distance from rotation center to detector

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_torch = torch.tensor(phantom, device=device, requires_grad=True)
angles_torch = torch.tensor(angles_np, device=device)

# Forward projection
sinogram = FanProjectorFunction.apply(image_torch, angles_torch, num_detectors,
                                      detector_spacing, source_distance, isocenter_distance)

# Fan beam FBP weighting and filtering
# Weight = cos(gamma), where gamma is the fan angle for each detector
u = (torch.arange(num_detectors, dtype=image_torch.dtype, device=device) - 
     (num_detectors - 1) / 2) * detector_spacing
gamma = torch.atan(u / source_distance)
weights = torch.cos(gamma).unsqueeze(0)  # Shape (1, num_detectors) for broadcasting

# Apply weights before filtering
sino_weighted = sinogram * weights
sinogram_filt = ramp_filter(sino_weighted).detach().requires_grad_(True).contiguous()

# Backprojection
reconstruction = FanBackprojectorFunction.apply(sinogram_filt, angles_torch,
                                                detector_spacing, Nx, Ny,
                                                source_distance, isocenter_distance)

# Fan beam FBP normalization
reconstruction = reconstruction * (math.pi / num_angles)

# Compute loss and gradients
loss = torch.mean((reconstruction - image_torch)**2)
loss.backward()

print("Loss:", loss.item())
print("Center pixel gradient:", image_torch.grad[Nx//2, Ny//2].item())
```

### Key Fan Beam Parameters Explained

#### Geometric Parameters
- **`source_distance`**: Distance from X-ray source to rotation center (typically 500-1000 units)
- **`isocenter_distance`**: Distance from rotation center to detector array (typically 300-600 units)
- **`detector_spacing`**: Physical spacing between detector elements
- **`num_detectors`**: Number of detector elements (often more than parallel beam due to magnification)

#### Parameter Selection Guidelines

| Parameter | Typical Range | Impact |
|-----------|---------------|---------|
| `source_distance` | 500-1000 | Larger = less magnification, smaller fan angle |
| `isocenter_distance` | 300-600 | Larger = more magnification, wider coverage |
| `num_detectors` | 400-800 | More detectors = higher resolution |
| `detector_spacing` | 0.5-2.0 | Affects sampling and field of view |

### Fan Beam Weighting

Fan beam FBP requires cosine weighting before filtering:

```python
# Calculate fan angles for each detector
u = (torch.arange(num_detectors) - (num_detectors - 1) / 2) * detector_spacing
gamma = torch.atan(u / source_distance)
weights = torch.cos(gamma)

# Apply weights to sinogram
sino_weighted = sinogram * weights.unsqueeze(0)
```

This weighting compensates for the varying path lengths in fan beam geometry.

### Magnification Factor

The magnification factor in fan beam geometry is:
```
M = (source_distance + isocenter_distance) / source_distance
```

This affects the effective field of view and resolution.

## Example 2: Fan Beam Iterative Reconstruction

Iterative reconstruction for fan beam geometry follows similar principles to parallel beam but uses the fan beam projector.

### Complete Code Example

```python
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import FanProjectorFunction

class FanIterativeRecoModel(nn.Module):
    """Neural network model for fan beam iterative reconstruction."""
    def __init__(self, volume_shape, angles, 
                 num_detectors, detector_spacing, 
                 source_distance, isocenter_distance):
        super().__init__()
        self.reco = nn.Parameter(torch.zeros(volume_shape))
        self.angles = angles
        self.num_detectors = num_detectors
        self.detector_spacing = detector_spacing
        self.source_distance = source_distance
        self.isocenter_distance = isocenter_distance

    def forward(self, x):
        updated_reco = x + self.reco
        current_sino = FanProjectorFunction.apply(
            updated_reco, self.angles, self.num_detectors, 
            self.detector_spacing, self.source_distance, self.isocenter_distance)
        return current_sino, updated_reco

class FanPipeline:
    """Training pipeline for fan beam iterative reconstruction."""
    def __init__(self, lr, volume_shape, angles, 
                 num_detectors, detector_spacing, 
                 source_distance, isocenter_distance, 
                 device, epoches=1000):
        self.epoches = epoches
        self.model = FanIterativeRecoModel(
            volume_shape, angles, num_detectors, detector_spacing, 
            source_distance, isocenter_distance).to(device)
        self.optimizer = optim.AdamW(list(self.model.parameters()), lr=lr)
        self.loss = nn.MSELoss()

    def train(self, input, label):
        loss_values = []
        for epoch in range(self.epoches):
            self.optimizer.zero_grad()
            predictions, current_reco = self.model(input)
            loss_value = self.loss(predictions, label)
            loss_value.backward()
            self.optimizer.step()
            loss_values.append(loss_value.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value.item()}")
                
        return loss_values, self.model

# Setup parameters for iterative reconstruction
Nx, Ny = 128, 128
phantom_cpu = shepp_logan_2d(Nx, Ny)
num_views = 360
angles_np = np.linspace(0, 2 * math.pi, num_views, endpoint=False).astype(np.float32)

# Fan beam geometry parameters
num_detectors = 256
detector_spacing = 0.75
source_distance = 600.0
isocenter_distance = 400.0

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
phantom_torch = torch.tensor(phantom_cpu, device=device)
angles_torch = torch.tensor(angles_np, device=device)

# Generate target sinogram
real_sinogram = FanProjectorFunction.apply(phantom_torch, angles_torch,
                                           num_detectors, detector_spacing,
                                           source_distance, isocenter_distance)

# Create and train reconstruction pipeline
pipeline_instance = FanPipeline(lr=1e-1,
                               volume_shape=(Ny, Nx),
                               angles=angles_torch,
                               num_detectors=num_detectors,
                               detector_spacing=detector_spacing,
                               source_distance=source_distance,
                               isocenter_distance=isocenter_distance,
                               device=device, epoches=1000)

ini_guess = torch.zeros_like(phantom_torch)
loss_values, trained_model = pipeline_instance.train(ini_guess, real_sinogram)

# Get final reconstruction
reco = trained_model(ini_guess)[1].squeeze().cpu().detach().numpy()
```

### Fan Beam vs Parallel Beam Differences

#### Geometric Complexity
- **Additional parameters**: Source and detector distances
- **Non-uniform sampling**: Varying magnification across field of view
- **Angular weighting**: Cosine weighting required for FBP

#### Computational Considerations
- **Memory usage**: Typically requires more detectors
- **Reconstruction quality**: Better dose efficiency than parallel beam
- **Artifacts**: Different artifact patterns (e.g., cone beam artifacts in 2D)

## Coordinate System Details

### Fan Beam Coordinate Convention

```
Source Position: (-source_distance, 0) relative to rotation center
Detector Array: (isocenter_distance, y) where y spans detector elements
Rotation Center: (0, 0) - origin of reconstruction coordinate system
```

### Detector Indexing

```python
# Detector positions relative to central ray
detector_positions = (torch.arange(num_detectors) - (num_detectors - 1) / 2) * detector_spacing

# Fan angles for each detector
fan_angles = torch.atan(detector_positions / source_distance)
```

### Field of View Calculation

The field of view (FOV) in fan beam geometry depends on the fan angle:

```python
max_fan_angle = torch.atan(((num_detectors - 1) / 2 * detector_spacing) / source_distance)
fov_diameter = 2 * source_distance * torch.tan(max_fan_angle)
```

## Visualization Examples

### Fan Beam Geometry Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_fan_geometry(source_distance, isocenter_distance, num_detectors, detector_spacing):
    """Visualize fan beam geometry setup."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Source position
    source_x = -source_distance
    source_y = 0
    
    # Detector positions
    detector_x = isocenter_distance
    detector_positions = (np.arange(num_detectors) - (num_detectors - 1) / 2) * detector_spacing
    
    # Plot source
    ax.plot(source_x, source_y, 'ro', markersize=10, label='X-ray Source')
    
    # Plot detector array
    ax.plot([detector_x] * num_detectors, detector_positions, 'b-', linewidth=3, label='Detector Array')
    
    # Plot fan beam rays (every 10th detector)
    for i in range(0, num_detectors, num_detectors//10):
        ax.plot([source_x, detector_x], [source_y, detector_positions[i]], 'g--', alpha=0.5)
    
    # Plot rotation center
    ax.plot(0, 0, 'ko', markersize=8, label='Rotation Center')
    
    # Add circle showing typical object size
    circle = plt.Circle((0, 0), min(source_distance, isocenter_distance) * 0.3, 
                       fill=False, linestyle='--', color='orange', label='Typical Object')
    ax.add_patch(circle)
    
    ax.set_xlim(-source_distance * 1.2, isocenter_distance * 1.2)
    ax.set_ylim(-detector_positions.max() * 1.2, detector_positions.max() * 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Fan Beam CT Geometry')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    plt.tight_layout()
    plt.show()

# Visualize geometry
visualize_fan_geometry(800, 500, 600, 1.0)
```

### Sinogram Comparison

```python
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(phantom, cmap='gray')
plt.title("Original Phantom")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sinogram.detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.title("Fan Beam Sinogram")
plt.xlabel("Detector Index")
plt.ylabel("Projection Angle")

plt.subplot(1, 3, 3)
plt.imshow(reconstruction.detach().cpu().numpy(), cmap='gray')
plt.title("Fan Beam Reconstruction")
plt.axis('off')

plt.tight_layout()
plt.show()
```

## Performance Optimization

### Memory Considerations

Fan beam reconstruction typically requires more memory due to:
- Larger detector arrays (higher magnification)
- Additional geometric parameters
- More complex interpolation operations

### GPU Acceleration Tips

```python
# Optimize tensor operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use appropriate data types
dtype = torch.float32  # Usually sufficient, faster than float64

# Batch operations when possible
angles_batch = angles_torch.unsqueeze(0)  # Add batch dimension if needed
```

### Parameter Tuning for Performance

```python
# Balance between quality and speed
performance_config = {
    'num_angles': 180,        # Minimum for good quality
    'num_detectors': 400,     # Adequate for most applications
    'detector_spacing': 1.0,  # Standard spacing
    'source_distance': 600,   # Moderate magnification
    'isocenter_distance': 400 # Good coverage
}

# High quality configuration
quality_config = {
    'num_angles': 720,        # High angular sampling
    'num_detectors': 800,     # High spatial resolution
    'detector_spacing': 0.5,  # Fine sampling
    'source_distance': 1000,  # Low magnification, less artifacts
    'isocenter_distance': 600 # Wide coverage
}
```

## Common Issues and Solutions

### Reconstruction Artifacts

#### Truncation Artifacts
- **Cause**: Object extends beyond field of view
- **Solution**: Increase `num_detectors` or adjust geometry parameters
- **Detection**: Check if object boundaries are visible in sinogram

#### Magnification Issues
- **Cause**: Incorrect geometric parameters
- **Solution**: Verify `source_distance` and `isocenter_distance` values
- **Check**: Magnification factor should be reasonable (1.5-3.0 typical)

#### Weighting Errors
- **Cause**: Missing or incorrect cosine weighting
- **Solution**: Always apply `cos(gamma)` weighting before filtering
- **Verification**: Check sinogram intensity distribution

### Training Issues

#### Slow Convergence
```python
# Try different learning rates
learning_rates = [1e-1, 5e-2, 1e-2]
for lr in learning_rates:
    # Test convergence speed
    pass
```

#### Memory Errors
```python
# Reduce problem size
reduced_config = {
    'Nx': 64, 'Ny': 64,           # Smaller image
    'num_detectors': 200,          # Fewer detectors
    'num_angles': 180              # Fewer angles
}
```

### Geometric Parameter Validation

```python
def validate_fan_geometry(source_distance, isocenter_distance, num_detectors, detector_spacing):
    """Validate fan beam geometry parameters."""
    
    # Check magnification factor
    magnification = (source_distance + isocenter_distance) / source_distance
    if magnification < 1.2 or magnification > 5.0:
        print(f"Warning: Unusual magnification factor {magnification:.2f}")
    
    # Check fan angle
    max_detector_pos = (num_detectors - 1) / 2 * detector_spacing
    max_fan_angle = np.arctan(max_detector_pos / source_distance)
    if max_fan_angle > np.pi / 3:  # 60 degrees
        print(f"Warning: Large fan angle {np.degrees(max_fan_angle):.1f}°")
    
    # Check field of view
    fov = 2 * source_distance * np.tan(max_fan_angle)
    print(f"Field of view: {fov:.1f} units")
    print(f"Magnification: {magnification:.2f}x")
    print(f"Max fan angle: {np.degrees(max_fan_angle):.1f}°")

# Validate your geometry
validate_fan_geometry(800, 500, 600, 1.0)
```

## Next Steps

After mastering fan beam reconstruction, consider:
- [Cone Beam Examples](cone_beam.md) for 3D reconstruction
- Advanced filtering techniques (Shepp-Logan, Hamming filters)
- Scatter correction methods
- Multi-energy CT reconstruction
- Real-time reconstruction optimization

## References and Further Reading

- Kak, A. C., & Slaney, M. (1988). Principles of computerized tomographic imaging
- Buzug, T. M. (2008). Computed tomography: from photon statistics to modern cone-beam CT
- Fan beam geometry specifications in medical CT standards