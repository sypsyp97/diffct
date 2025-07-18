# Parallel Beam Examples

This section demonstrates how to use DiffCT for parallel beam CT reconstruction using both Filtered Back Projection (FBP) and iterative reconstruction methods.

## Overview

Parallel beam geometry is the simplest CT geometry where X-rays travel in parallel lines through the object. This geometry is commonly used in synchrotron CT and some medical CT scanners. DiffCT provides differentiable operators for both forward projection and backprojection in parallel beam geometry.

## Basic Setup

All parallel beam examples use the following key components:

- `ParallelProjectorFunction`: Forward projection (image → sinogram)
- `ParallelBackprojectorFunction`: Backprojection (sinogram → image)
- Shepp-Logan phantom for testing
- PyTorch tensors with automatic differentiation support

## Example 1: Filtered Back Projection (FBP)

FBP is the standard analytical reconstruction method for CT. This example shows how to implement differentiable FBP using DiffCT.

### Complete Code Example

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffct.differentiable import ParallelProjectorFunction, ParallelBackprojectorFunction

def shepp_logan_2d(Nx, Ny):
    """Generate 2D Shepp-Logan phantom for testing."""
    Nx = int(Nx)
    Ny = int(Ny)
    phantom = np.zeros((Nx, Ny), dtype=np.float64)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 1.0),
        (0.0, -0.0184, 0.6624, 0.8740, 0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18.0, -0.8),
        (-0.22, 0.0, 0.16, 0.41, 18.0, -0.8),
        (0.0, 0.35, 0.21, 0.25, 0, 0.7),
    ]
    cx = (Nx - 1) / 2
    cy = (Ny - 1) / 2
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

# Setup parameters
Nx, Ny = 256, 256
phantom = shepp_logan_2d(Nx, Ny)
num_angles = 360
angles_np = np.linspace(0, 2*np.pi, num_angles, endpoint=False).astype(np.float32)
num_detectors = 512
detector_spacing = 1.0

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_torch = torch.tensor(phantom, device=device, requires_grad=True)
angles_torch = torch.tensor(angles_np, device=device, requires_grad=False)

# Forward projection
sinogram = ParallelProjectorFunction.apply(image_torch, angles_torch,
                                           num_detectors, detector_spacing)

# Apply ramp filter
sinogram_filt = ramp_filter(sinogram).detach().requires_grad_(True).contiguous()

# Backprojection
reconstruction = ParallelBackprojectorFunction.apply(sinogram_filt, angles_torch,
                                                     detector_spacing, Nx, Ny)

# FBP normalization
reconstruction = reconstruction * (np.pi / num_angles)

# Compute loss and gradients
loss = torch.mean((reconstruction - image_torch)**2)
loss.backward()

print("Loss:", loss.item())
print("Phantom gradient center pixel:", image_torch.grad[Nx//2, Ny//2].item())
```

### Key Parameters Explained

- **`num_angles`**: Number of projection angles (360 for full rotation)
- **`angles_np`**: Projection angles in radians, typically from 0 to 2π
- **`num_detectors`**: Number of detector elements (512 provides good resolution)
- **`detector_spacing`**: Physical spacing between detector elements
- **`requires_grad=True`**: Enables automatic differentiation for the image

### FBP Normalization

The FBP reconstruction requires proper normalization:
```python
reconstruction = reconstruction * (np.pi / num_angles)
```

This accounts for:
- Angular step size: `2π / num_angles`
- FBP integral normalization factor: `1/2`
- Combined factor: `π / num_angles`

### Expected Output

The FBP reconstruction should closely match the original phantom with:
- Sharp edges preserved
- Correct intensity values
- Minimal artifacts for sufficient angular sampling

## Example 2: Iterative Reconstruction

Iterative reconstruction uses gradient-based optimization to minimize the difference between measured and computed projections.

### Complete Code Example

```python
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import ParallelProjectorFunction

class IterativeRecoModel(nn.Module):
    """Neural network model for iterative reconstruction."""
    def __init__(self, volume_shape, angles, num_detectors, detector_spacing):
        super().__init__()
        self.reco = nn.Parameter(torch.zeros(volume_shape))
        self.angles = angles
        self.num_detectors = num_detectors
        self.detector_spacing = detector_spacing

    def forward(self, x):
        updated_reco = x + self.reco
        current_sino = ParallelProjectorFunction.apply(updated_reco, self.angles, 
                                                       self.num_detectors, self.detector_spacing)
        return current_sino, updated_reco

class Pipeline:
    """Training pipeline for iterative reconstruction."""
    def __init__(self, lr, volume_shape, angles, num_detectors, detector_spacing, 
                 device, epoches=1000):
        self.epoches = epoches
        self.model = IterativeRecoModel(volume_shape, angles, num_detectors, 
                                        detector_spacing).to(device)
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

# Setup parameters
Nx, Ny = 128, 128
phantom_cpu = shepp_logan_2d(Nx, Ny)
num_views = 360
angles_np = np.linspace(0, 2 * math.pi, num_views, endpoint=False).astype(np.float32)
num_detectors = 256
detector_spacing = 0.75

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
phantom_torch = torch.tensor(phantom_cpu, device=device)
angles_torch = torch.tensor(angles_np, device=device)

# Generate target sinogram
real_sinogram = ParallelProjectorFunction.apply(phantom_torch, angles_torch,
                                                num_detectors, detector_spacing)

# Create and train reconstruction pipeline
pipeline_instance = Pipeline(lr=1e-1,
                             volume_shape=(Ny, Nx),
                             angles=angles_torch,
                             num_detectors=num_detectors,
                             detector_spacing=detector_spacing,
                             device=device, epoches=1000)

ini_guess = torch.zeros_like(phantom_torch)
loss_values, trained_model = pipeline_instance.train(ini_guess, real_sinogram)

# Get final reconstruction
reco = trained_model(ini_guess)[1].squeeze().cpu().detach().numpy()
```

### Key Components Explained

#### IterativeRecoModel
- **`nn.Parameter`**: Makes the reconstruction image a trainable parameter
- **Forward method**: Computes forward projection of current reconstruction
- **Residual connection**: `x + self.reco` allows for flexible initialization

#### Training Pipeline
- **AdamW optimizer**: Adaptive learning rate with weight decay
- **MSE Loss**: Minimizes squared difference between projections
- **Learning rate**: `1e-1` works well for most cases

### Optimization Parameters

- **Learning rate**: Start with `1e-1`, reduce if training is unstable
- **Epochs**: 1000 epochs usually sufficient for convergence
- **Batch size**: Single image reconstruction (batch size = 1)
- **Optimizer**: AdamW provides good convergence properties

### Expected Convergence

The loss should decrease steadily:
- Initial loss: High (depends on phantom intensity)
- Final loss: Near zero for noise-free data
- Convergence: Typically within 100-500 epochs

## Visualization Examples

### Displaying Results

```python
import matplotlib.pyplot as plt

# Create comparison plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(phantom, cmap='gray')
plt.title("Original Phantom")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sinogram.detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.title("Sinogram")
plt.xlabel("Detector Index")
plt.ylabel("Projection Angle")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(reconstruction.detach().cpu().numpy(), cmap='gray')
plt.title("Reconstruction")
plt.axis('off')

plt.tight_layout()
plt.show()
```

### Loss Curve Visualization

```python
plt.figure(figsize=(10, 6))
plt.plot(loss_values)
plt.title("Iterative Reconstruction Convergence")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.yscale('log')
plt.grid(True)
plt.show()
```

## Performance Considerations

### GPU Acceleration
- Use CUDA when available: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Larger images benefit more from GPU acceleration
- Memory usage scales with image size and number of projections

### Parameter Selection Guidelines

| Parameter | Typical Range | Impact |
|-----------|---------------|---------|
| `num_angles` | 180-720 | More angles = better quality, slower computation |
| `num_detectors` | 256-1024 | More detectors = higher resolution |
| `detector_spacing` | 0.5-2.0 | Affects field of view and sampling |
| Learning rate | 1e-2 to 1e-1 | Higher = faster convergence, risk of instability |

### Memory Usage
- Image size: 256×256 uses ~1MB per image
- Sinogram size: 360×512 uses ~1.4MB
- GPU memory scales linearly with problem size

## Common Issues and Solutions

### Reconstruction Artifacts
- **Streaking**: Insufficient angular sampling → increase `num_angles`
- **Blurring**: Poor detector sampling → increase `num_detectors`
- **Ring artifacts**: Detector calibration issues → check `detector_spacing`

### Training Issues
- **Slow convergence**: Reduce learning rate or increase epochs
- **Instability**: Lower learning rate, check gradient clipping
- **Memory errors**: Reduce image size or use CPU

### Gradient Flow
- Always use `requires_grad=True` for parameters you want to optimize
- Use `.detach()` when breaking gradient flow is needed
- Check gradient magnitudes with `tensor.grad.norm()`

## Next Steps

After mastering parallel beam reconstruction, consider:
- [Fan Beam Examples](fan_beam.md) for more realistic geometries
- [Cone Beam Examples](cone_beam.md) for 3D reconstruction
- Advanced regularization techniques
- Multi-GPU training for large problems