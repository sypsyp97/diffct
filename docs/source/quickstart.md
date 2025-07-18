# Quick Start Guide

This guide will help you get up and running with DiffCT quickly. Follow these steps to install the library and run your first CT reconstruction examples.

## Prerequisites

Before installing DiffCT, ensure you have the following prerequisites:

### Hardware Requirements
- **CUDA-capable GPU**: Required for GPU acceleration
- **Minimum 4GB GPU memory**: Recommended for basic examples
- **8GB+ GPU memory**: Recommended for larger reconstructions

### Software Requirements
- **Python 3.10+**: DiffCT requires Python 3.10 or later
- **CUDA Toolkit**: Compatible CUDA installation
- **PyTorch with CUDA support**: For tensor operations and automatic differentiation

## Installation

### Option 1: Install from PyPI (Recommended)

The easiest way to install DiffCT is using pip:

```bash
pip install diffct
```

### Option 2: Install from Source

For the latest development version or to contribute:

```bash
# Clone the repository
git clone https://github.com/sypsyp97/diffct.git
cd diffct

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .
```

### Setting up CUDA Environment

If you don't have PyTorch with CUDA support, install it first:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation

Test your installation with this simple script:

```python
import torch
import diffct

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"DiffCT version: {diffct.__version__}")

# Test basic import
from diffct import ParallelProjectorFunction
print("DiffCT installed successfully!")
```

## Basic Usage Examples

### 1. Parallel Beam Geometry

Parallel beam geometry is the simplest CT geometry, where X-rays are parallel to each other. For more detailed examples, see {doc}`examples/parallel_beam`.

```python
import torch
import numpy as np
from diffct import ParallelProjectorFunction, ParallelBackprojectorFunction

# Create a simple 2D phantom
phantom = torch.randn(256, 256, device='cuda', requires_grad=True)

# Define projection parameters
angles = torch.linspace(0, 2*np.pi, 180, device='cuda')  # 180 projection angles
num_detectors = 512                                       # Number of detector elements
detector_spacing = 1.0                                    # Spacing between detectors

# Forward projection (image → sinogram)
sinogram = ParallelProjectorFunction.apply(
    phantom, angles, num_detectors, detector_spacing
)

# Backprojection (sinogram → image)
reconstruction = ParallelBackprojectorFunction.apply(
    sinogram, angles, detector_spacing, 256, 256  # Output image dimensions
)

print(f"Sinogram shape: {sinogram.shape}")        # [180, 512]
print(f"Reconstruction shape: {reconstruction.shape}")  # [256, 256]
```

### 2. Fan Beam Geometry

Fan beam geometry uses a point X-ray source with a fan-shaped beam. For comprehensive fan beam examples, see {doc}`examples/fan_beam`.

```python
import torch
import numpy as np
from diffct import FanProjectorFunction, FanBackprojectorFunction

# Create a 2D phantom
phantom = torch.randn(256, 256, device='cuda', requires_grad=True)

# Define fan beam parameters
angles = torch.linspace(0, 2*np.pi, 360, device='cuda')
num_detectors = 600
detector_spacing = 1.0
source_distance = 800.0      # Distance from source to rotation center
isocenter_distance = 500.0   # Distance from rotation center to detector

# Forward projection
sinogram = FanProjectorFunction.apply(
    phantom, angles, num_detectors, detector_spacing,
    source_distance, isocenter_distance
)

# Backprojection
reconstruction = FanBackprojectorFunction.apply(
    sinogram, angles, detector_spacing, 256, 256,
    source_distance, isocenter_distance
)

print(f"Fan beam sinogram shape: {sinogram.shape}")
```

### 3. Cone Beam Geometry

Cone beam geometry extends fan beam to 3D with a cone-shaped X-ray beam. For detailed 3D reconstruction examples, see {doc}`examples/cone_beam`.

```python
import torch
import numpy as np
from diffct import ConeProjectorFunction, ConeBackprojectorFunction

# Create a 3D phantom
phantom = torch.randn(128, 128, 128, device='cuda', requires_grad=True)

# Define cone beam parameters
angles = torch.linspace(0, 2*np.pi, 360, device='cuda')
det_u, det_v = 256, 256      # Detector dimensions (horizontal, vertical)
du, dv = 1.0, 1.0           # Detector pixel spacing
source_distance = 900.0      # Source to rotation center distance
isocenter_distance = 600.0   # Rotation center to detector distance

# Forward projection
sinogram = ConeProjectorFunction.apply(
    phantom, angles, det_u, det_v, du, dv,
    source_distance, isocenter_distance
)

# Backprojection
reconstruction = ConeBackprojectorFunction.apply(
    sinogram, angles, 128, 128, 128,  # Output volume dimensions
    du, dv, source_distance, isocenter_distance
)

print(f"Cone beam sinogram shape: {sinogram.shape}")  # [360, 256, 256]
print(f"3D reconstruction shape: {reconstruction.shape}")  # [128, 128, 128]
```

## Complete Working Example

Here's a complete example showing FBP (Filtered Backprojection) reconstruction:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffct import ParallelProjectorFunction, ParallelBackprojectorFunction

def create_phantom(size=256):
    """Create a simple circular phantom"""
    phantom = torch.zeros(size, size)
    center = size // 2
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    
    # Create circles with different intensities
    mask1 = (x - center)**2 + (y - center)**2 < (size//4)**2
    mask2 = (x - center)**2 + (y - center)**2 < (size//8)**2
    
    phantom[mask1] = 0.5
    phantom[mask2] = 1.0
    
    return phantom

def ramp_filter(sinogram):
    """Apply ramp filter for FBP reconstruction"""
    device = sinogram.device
    num_views, num_det = sinogram.shape
    
    # Create ramp filter in frequency domain
    freqs = torch.fft.fftfreq(num_det, device=device)
    ramp = torch.abs(2 * torch.pi * freqs)
    
    # Apply filter
    sino_fft = torch.fft.fft(sinogram, dim=1)
    filtered_fft = sino_fft * ramp.unsqueeze(0)
    filtered = torch.real(torch.fft.ifft(filtered_fft, dim=1))
    
    return filtered

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom = create_phantom().to(device)
    
    # Projection parameters
    num_angles = 180
    angles = torch.linspace(0, np.pi, num_angles, device=device)
    num_detectors = 512
    detector_spacing = 1.0
    
    # Forward projection
    sinogram = ParallelProjectorFunction.apply(
        phantom, angles, num_detectors, detector_spacing
    )
    
    # FBP reconstruction
    filtered_sinogram = ramp_filter(sinogram)
    reconstruction = ParallelBackprojectorFunction.apply(
        filtered_sinogram, angles, detector_spacing, 256, 256
    )
    
    # Normalize reconstruction
    reconstruction *= (np.pi / num_angles)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(phantom.cpu(), cmap='gray')
    axes[0].set_title('Original Phantom')
    axes[0].axis('off')
    
    axes[1].imshow(sinogram.cpu(), cmap='gray', aspect='auto')
    axes[1].set_title('Sinogram')
    axes[1].axis('off')
    
    axes[2].imshow(reconstruction.cpu(), cmap='gray')
    axes[2].set_title('FBP Reconstruction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Reconstruction error: {torch.mean((reconstruction - phantom)**2).item():.6f}")

if __name__ == "__main__":
    main()
```

## Gradient-Based Optimization Example

DiffCT's differentiable operators enable gradient-based optimization:

```python
import torch
import torch.optim as optim
from diffct import ParallelProjectorFunction

# Create target sinogram from known phantom
target_phantom = torch.randn(128, 128, device='cuda')
angles = torch.linspace(0, 2*np.pi, 180, device='cuda')

with torch.no_grad():
    target_sinogram = ParallelProjectorFunction.apply(
        target_phantom, angles, 256, 1.0
    )

# Initialize reconstruction with zeros
reconstruction = torch.zeros_like(target_phantom, requires_grad=True)
optimizer = optim.Adam([reconstruction], lr=0.01)

# Optimization loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward projection of current reconstruction
    predicted_sinogram = ParallelProjectorFunction.apply(
        reconstruction, angles, 256, 1.0
    )
    
    # Compute loss
    loss = torch.mean((predicted_sinogram - target_sinogram)**2)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("Optimization completed!")
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce image/volume size
- Decrease number of projection angles
- Use smaller batch sizes

**Import Errors**
```bash
# Ensure all dependencies are installed
pip install torch numpy numba

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Slow Performance**
- Ensure you're using GPU (`device='cuda'`)
- Check CUDA toolkit compatibility
- Monitor GPU memory usage

### Performance Tips

1. **Use appropriate data types**: `torch.float32` is usually sufficient
2. **Batch operations**: Process multiple projections together when possible
3. **Memory management**: Use `torch.cuda.empty_cache()` to free GPU memory
4. **Tensor contiguity**: Ensure tensors are contiguous with `.contiguous()`

### Getting Help

- **GitHub Issues**: [Report bugs or ask questions](https://github.com/sypsyp97/diffct/issues)
- **Documentation**: Check the API reference for detailed parameter descriptions
- **Examples**: Explore the `examples/` directory for more use cases

## Next Steps

Now that you have DiffCT running, explore:

- {doc}`api/index`: Detailed function documentation and API reference
- {doc}`examples/index`: Advanced usage patterns and comprehensive tutorials
- {doc}`api/geometries`: Understanding CT geometries and parameter configurations
- {doc}`examples/parallel_beam`: Start with parallel beam reconstruction examples
- {doc}`examples/fan_beam`: Learn fan beam geometry and applications
- {doc}`examples/cone_beam`: Explore 3D cone beam reconstruction

### Recommended Learning Path

1. **Beginners**: Start with {doc}`examples/parallel_beam` for fundamental concepts
2. **Intermediate**: Move to {doc}`examples/fan_beam` for clinical applications  
3. **Advanced**: Explore {doc}`examples/cone_beam` for 3D volumetric reconstruction
4. **Deep Learning**: All examples include gradient computation for ML integration

### Additional Resources

- **GitHub Repository**: [Source code and additional examples](https://github.com/sypsyp97/diffct)
- **Issue Tracker**: [Report bugs or request features](https://github.com/sypsyp97/diffct/issues)
- **API Documentation**: {doc}`api/projectors` for detailed function signatures

Happy reconstructing! 🚀