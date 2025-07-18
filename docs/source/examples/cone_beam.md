# Cone Beam Examples

This section demonstrates how to use DiffCT for 3D cone beam CT reconstruction using both FDK (Feldkamp-Davis-Kress) and iterative reconstruction methods. Cone beam geometry extends fan beam to 3D and is widely used in medical imaging, industrial CT, and micro-CT applications.

## Overview

Cone beam CT uses a 2D detector array and a point X-ray source to acquire 3D volumetric data in a single rotation. This geometry provides excellent dose efficiency and fast acquisition times, making it ideal for real-time imaging applications. DiffCT provides differentiable operators for both forward projection and backprojection in cone beam geometry.

## Cone Beam Geometry Fundamentals

### 3D Coordinate System

In cone beam geometry, the key components are:

- **Point X-ray Source**: Located at distance R from rotation center
- **2D Detector Array**: Flat panel detector at distance D from rotation center
- **Volume of Interest**: 3D object centered at rotation axis
- **Cone Angle**: 3D cone of X-rays from source through volume to detector

### Geometric Parameters

```python
# Core geometry parameters
source_distance = 900.0      # Distance from source to rotation center (R)
isocenter_distance = 600.0   # Distance from rotation center to detector (D)
det_u, det_v = 256, 256     # Detector array dimensions (horizontal, vertical)
du, dv = 1.0, 1.0           # Detector pixel spacing
Nx, Ny, Nz = 128, 128, 128  # Volume dimensions
```

### Coordinate Conventions

```
Source Position: (-source_distance, 0, 0) relative to rotation center
Detector Center: (isocenter_distance, 0, 0) relative to rotation center
Volume Center: (0, 0, 0) - rotation center
Detector Coordinates: (u, v) where u is horizontal, v is vertical
```

## Example 1: FDK Cone Beam Reconstruction

The FDK algorithm is the standard analytical reconstruction method for cone beam CT, extending the fan beam FBP algorithm to 3D.

### Complete Code Example

```python
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffct.differentiable import ConeProjectorFunction, ConeBackprojectorFunction

def shepp_logan_3d(shape):
    """Generate 3D Shepp-Logan phantom for testing."""
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    xx = (xx - (shape[2] - 1) / 2) / ((shape[2] - 1) / 2)
    yy = (yy - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    zz = (zz - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)
    
    # 3D ellipsoid parameters: [x, y, z, a, b, c, phi, theta, psi, intensity]
    el_params = np.array([
        [0, 0, 0, 0.69, 0.92, 0.81, 0, 0, 0, 1],
        [0, -0.0184, 0, 0.6624, 0.874, 0.78, 0, 0, 0, -0.8],
        [0.22, 0, 0, 0.11, 0.31, 0.22, -np.pi/10.0, 0, 0, -0.2],
        [-0.22, 0, 0, 0.16, 0.41, 0.28, np.pi/10.0, 0, 0, -0.2],
        [0, 0.35, -0.15, 0.21, 0.25, 0.41, 0, 0, 0, 0.1],
        [0, 0.1, 0.25, 0.046, 0.046, 0.05, 0, 0, 0, 0.1],
        [0, -0.1, 0.25, 0.046, 0.046, 0.05, 0, 0, 0, 0.1],
        [-0.08, -0.605, 0, 0.046, 0.023, 0.05, 0, 0, 0, 0.1],
        [0, -0.605, 0, 0.023, 0.023, 0.02, 0, 0, 0, 0.1],
        [0.06, -0.605, 0, 0.023, 0.046, 0.02, 0, 0, 0, 0.1],
    ], dtype=np.float32)

    # Extract parameters for vectorized computation
    x_pos = el_params[:, 0][:, None, None, None]
    y_pos = el_params[:, 1][:, None, None, None]
    z_pos = el_params[:, 2][:, None, None, None]
    a_axis = el_params[:, 3][:, None, None, None]
    b_axis = el_params[:, 4][:, None, None, None]
    c_axis = el_params[:, 5][:, None, None, None]
    phi = el_params[:, 6][:, None, None, None]
    val = el_params[:, 9][:, None, None, None]

    # Translate coordinates to ellipsoid centers
    xc = xx[None, ...] - x_pos
    yc = yy[None, ...] - y_pos
    zc = zz[None, ...] - z_pos

    # Apply rotation (simplified - only z-axis rotation)
    c = np.cos(phi)
    s = np.sin(phi)
    xp = c * xc - s * yc
    yp = s * xc + c * yc
    zp = zc

    # Check if points are inside ellipsoids
    mask = (
        (xp ** 2) / (a_axis ** 2)
        + (yp ** 2) / (b_axis ** 2)
        + (zp ** 2) / (c_axis ** 2)
        <= 1.0
    )

    # Sum contributions from all ellipsoids
    shepp_logan = np.sum(mask * val, axis=0)
    shepp_logan = np.clip(shepp_logan, 0, 1)
    return shepp_logan

def ramp_filter_3d(sinogram_tensor):
    """Apply ramp filter for FDK reconstruction."""
    device = sinogram_tensor.device
    num_views, num_det_u, num_det_v = sinogram_tensor.shape
    freqs = torch.fft.fftfreq(num_det_u, device=device)
    omega = 2.0 * torch.pi * freqs
    ramp = torch.abs(omega)
    ramp_3d = ramp.reshape(1, num_det_u, 1)
    sino_fft = torch.fft.fft(sinogram_tensor, dim=1)
    filtered_fft = sino_fft * ramp_3d
    filtered = torch.real(torch.fft.ifft(filtered_fft, dim=1))
    return filtered

# Setup 3D cone beam geometry
Nx, Ny, Nz = 128, 128, 128
phantom_cpu = shepp_logan_3d((Nx, Ny, Nz))
num_views = 360
angles_np = np.linspace(0, 2*math.pi, num_views, endpoint=False).astype(np.float32)

# Cone beam specific parameters
det_u, det_v = 256, 256      # 2D detector dimensions
du, dv = 1.0, 1.0           # Detector pixel spacing
source_distance = 900.0      # Source to rotation center
isocenter_distance = 600.0   # Rotation center to detector

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
phantom_torch = torch.tensor(phantom_cpu, device=device, requires_grad=True)
angles_torch = torch.tensor(angles_np, device=device)

# Forward projection
sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch,
                                       det_u, det_v, du, dv,
                                       source_distance, isocenter_distance)

# FDK weighting and filtering
# Weight = D / sqrt(D^2 + u^2 + v^2), where D is source_distance
u_coords = (torch.arange(det_u, dtype=phantom_torch.dtype, device=device) - 
           (det_u - 1) / 2) * du
v_coords = (torch.arange(det_v, dtype=phantom_torch.dtype, device=device) - 
           (det_v - 1) / 2) * dv

# Reshape for broadcasting over sinogram of shape (views, u, v)
u_coords = u_coords.view(1, det_u, 1)
v_coords = v_coords.view(1, 1, det_v)

weights = source_distance / torch.sqrt(source_distance**2 + u_coords**2 + v_coords**2)

# Apply weights and filter
sino_weighted = sinogram * weights
sinogram_filt = ramp_filter_3d(sino_weighted).detach().requires_grad_(True).contiguous()

# Backprojection
reconstruction = ConeBackprojectorFunction.apply(sinogram_filt, angles_torch, Nx, Ny, Nz,
                                                du, dv, source_distance, isocenter_distance)

# FDK normalization
reconstruction = reconstruction * (math.pi / num_views)

# Compute loss and gradients
loss = torch.mean((reconstruction - phantom_torch)**2)
loss.backward()

print("Cone Beam FDK Reconstruction:")
print("Loss:", loss.item())
print("Volume center voxel gradient:", phantom_torch.grad[Nx//2, Ny//2, Nz//2].item())
print("Reconstruction shape:", reconstruction.shape)
```

### Key FDK Components Explained

#### 3D Weighting Function
The FDK algorithm requires distance-dependent weighting:

```python
# Distance from source to each detector pixel
distance = torch.sqrt(source_distance**2 + u_coords**2 + v_coords**2)
weights = source_distance / distance
```

This weighting compensates for the varying path lengths in the cone beam geometry.

#### 3D Ramp Filtering
Filtering is applied only in the u-direction (horizontal):

```python
def ramp_filter_3d(sinogram_tensor):
    # Filter along u-direction only (dim=1)
    sino_fft = torch.fft.fft(sinogram_tensor, dim=1)
    filtered_fft = sino_fft * ramp_filter
    return torch.real(torch.fft.ifft(filtered_fft, dim=1))
```

#### Volume Reconstruction
The 3D backprojection accumulates filtered projections:

```python
reconstruction = ConeBackprojectorFunction.apply(
    sinogram_filt, angles_torch, Nx, Ny, Nz,
    du, dv, source_distance, isocenter_distance)
```

## Example 2: Cone Beam Iterative Reconstruction

Iterative reconstruction for cone beam geometry provides better image quality, especially for limited-angle or sparse-view scenarios.

### Complete Code Example

```python
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import ConeProjectorFunction

class ConeIterativeRecoModel(nn.Module):
    """Neural network model for cone beam iterative reconstruction."""
    def __init__(self, volume_shape, angles, det_u, det_v, du, dv, 
                 source_distance, isocenter_distance):
        super().__init__()
        self.reco = nn.Parameter(torch.zeros(volume_shape))
        self.angles = angles
        self.det_u = det_u
        self.det_v = det_v
        self.du = du
        self.dv = dv
        self.source_distance = source_distance
        self.isocenter_distance = isocenter_distance

    def forward(self, x):
        updated_reco = x + self.reco
        current_sino = ConeProjectorFunction.apply(
            updated_reco, self.angles, 
            self.det_u, self.det_v, self.du, self.dv, 
            self.source_distance, self.isocenter_distance)
        return current_sino, updated_reco

class ConePipeline:
    """Training pipeline for cone beam iterative reconstruction."""
    def __init__(self, lr, volume_shape, angles, 
                 det_u, det_v, du, dv, 
                 source_distance, isocenter_distance, 
                 device, epoches=1000):
        self.epoches = epoches
        self.model = ConeIterativeRecoModel(
            volume_shape, angles, det_u, det_v, du, dv, 
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
Nx, Ny, Nz = 64, 64, 64  # Smaller volume for faster computation
phantom_cpu = shepp_logan_3d((Nx, Ny, Nz))
num_views = 180  # Reduced views for faster computation
angles_np = np.linspace(0, 2 * math.pi, num_views, endpoint=False).astype(np.float32)

# Cone beam geometry parameters
det_u, det_v = 128, 128
du, dv = 1.0, 1.0
source_distance = 600.0
isocenter_distance = 400.0

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
phantom_torch = torch.tensor(phantom_cpu, device=device)
angles_torch = torch.tensor(angles_np, device=device)

# Generate target sinogram
real_sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch,
                                           det_u, det_v, du, dv,
                                           source_distance, isocenter_distance)

# Create and train reconstruction pipeline
pipeline_instance = ConePipeline(lr=1e-1, 
                                volume_shape=(Nz, Ny, Nx), 
                                angles=angles_torch, 
                                det_u=det_u, det_v=det_v, 
                                du=du, dv=dv, 
                                source_distance=source_distance, 
                                isocenter_distance=isocenter_distance, 
                                device=device, epoches=1000)

ini_guess = torch.zeros_like(phantom_torch)
loss_values, trained_model = pipeline_instance.train(ini_guess, real_sinogram)

# Get final reconstruction
reco = trained_model(ini_guess)[1].squeeze().cpu().detach().numpy()
```

### 3D Reconstruction Considerations

#### Memory Management
3D reconstruction requires significantly more memory:

```python
# Memory usage estimation
volume_memory = Nx * Ny * Nz * 4  # bytes for float32
sinogram_memory = num_views * det_u * det_v * 4  # bytes for float32
total_memory_mb = (volume_memory + sinogram_memory) / (1024**2)

print(f"Estimated memory usage: {total_memory_mb:.1f} MB")
```

#### Computational Complexity
- **Forward projection**: O(Nx × Ny × Nz × num_views × det_u × det_v)
- **Backprojection**: O(Nx × Ny × Nz × num_views × det_u × det_v)
- **Memory scaling**: Cubic with volume size, quadratic with detector size

## 3D Visualization Examples

### Multi-Slice Visualization

```python
import matplotlib.pyplot as plt

def visualize_3d_volume(volume, title="3D Volume"):
    """Visualize 3D volume with multiple slice views."""
    Nz, Ny, Nx = volume.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Axial slices (XY planes)
    for i, z_idx in enumerate([Nz//4, Nz//2, 3*Nz//4]):
        axes[0, i].imshow(volume[z_idx, :, :], cmap='gray')
        axes[0, i].set_title(f'Axial Slice Z={z_idx}')
        axes[0, i].axis('off')
    
    # Sagittal slices (YZ planes)
    for i, x_idx in enumerate([Nx//4, Nx//2, 3*Nx//4]):
        axes[1, i].imshow(volume[:, :, x_idx], cmap='gray')
        axes[1, i].set_title(f'Sagittal Slice X={x_idx}')
        axes[1, i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Visualize phantom and reconstruction
visualize_3d_volume(phantom_cpu, "Original 3D Phantom")
visualize_3d_volume(reconstruction.detach().cpu().numpy(), "3D Reconstruction")
```

### Projection Visualization

```python
def visualize_cone_projections(sinogram, num_views_to_show=4):
    """Visualize cone beam projections."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    view_indices = np.linspace(0, sinogram.shape[0]-1, num_views_to_show, dtype=int)
    
    for i, view_idx in enumerate(view_indices):
        axes[i].imshow(sinogram[view_idx], cmap='gray', aspect='auto')
        axes[i].set_title(f'Projection View {view_idx}')
        axes[i].set_xlabel('Detector U')
        axes[i].set_ylabel('Detector V')
    
    plt.tight_layout()
    plt.show()

# Visualize projections
visualize_cone_projections(sinogram.detach().cpu().numpy())
```

### 3D Geometry Visualization

```python
def visualize_cone_geometry(source_distance, isocenter_distance, det_u, det_v, du, dv):
    """Visualize cone beam geometry setup."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Source position
    source_pos = np.array([-source_distance, 0, 0])
    ax.scatter(*source_pos, color='red', s=100, label='X-ray Source')
    
    # Detector corners
    u_max = (det_u - 1) / 2 * du
    v_max = (det_v - 1) / 2 * dv
    detector_corners = np.array([
        [isocenter_distance, -u_max, -v_max],
        [isocenter_distance, u_max, -v_max],
        [isocenter_distance, u_max, v_max],
        [isocenter_distance, -u_max, v_max],
        [isocenter_distance, -u_max, -v_max]  # Close the rectangle
    ])
    
    # Plot detector
    ax.plot(detector_corners[:, 0], detector_corners[:, 1], detector_corners[:, 2], 
            'b-', linewidth=3, label='Detector Array')
    
    # Plot cone rays to detector corners
    for corner in detector_corners[:-1]:  # Exclude duplicate point
        ax.plot([source_pos[0], corner[0]], 
                [source_pos[1], corner[1]], 
                [source_pos[2], corner[2]], 'g--', alpha=0.5)
    
    # Plot rotation center
    ax.scatter(0, 0, 0, color='black', s=50, label='Rotation Center')
    
    # Add volume outline
    vol_size = min(source_distance, isocenter_distance) * 0.3
    volume_corners = np.array([
        [-vol_size, -vol_size, -vol_size],
        [vol_size, -vol_size, -vol_size],
        [vol_size, vol_size, -vol_size],
        [-vol_size, vol_size, -vol_size],
        [-vol_size, -vol_size, vol_size],
        [vol_size, -vol_size, vol_size],
        [vol_size, vol_size, vol_size],
        [-vol_size, vol_size, vol_size]
    ])
    
    # Draw volume edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    for edge in edges:
        points = volume_corners[edge]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'orange', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Cone Beam CT Geometry')
    plt.show()

# Visualize geometry
visualize_cone_geometry(900, 600, 256, 256, 1.0, 1.0)
```

## Performance Optimization for 3D

### Memory Optimization Strategies

```python
# Strategy 1: Reduce precision
phantom_torch = phantom_torch.half()  # Use float16 instead of float32

# Strategy 2: Process in chunks
def process_volume_chunks(volume, chunk_size=32):
    """Process volume in smaller chunks to reduce memory usage."""
    Nz, Ny, Nx = volume.shape
    results = []
    
    for z_start in range(0, Nz, chunk_size):
        z_end = min(z_start + chunk_size, Nz)
        chunk = volume[z_start:z_end]
        # Process chunk
        result_chunk = process_chunk(chunk)
        results.append(result_chunk)
    
    return torch.cat(results, dim=0)

# Strategy 3: Gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, input):
    return checkpoint(model, input)
```

### GPU Memory Management

```python
# Monitor GPU memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

# Clear cache when needed
torch.cuda.empty_cache()

# Use memory-efficient operations
with torch.cuda.amp.autocast():  # Automatic mixed precision
    sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch, ...)
```

### Computational Optimization

```python
# Optimize for different problem sizes
def get_optimal_config(target_memory_gb=8):
    """Get optimal configuration based on available memory."""
    configs = [
        {'volume': (64, 64, 64), 'detector': (128, 128), 'views': 180},    # Low memory
        {'volume': (128, 128, 128), 'detector': (256, 256), 'views': 360}, # Medium memory
        {'volume': (256, 256, 256), 'detector': (512, 512), 'views': 720}  # High memory
    ]
    
    for config in configs:
        estimated_memory = estimate_memory_usage(config)
        if estimated_memory <= target_memory_gb:
            return config
    
    return configs[0]  # Return smallest config if none fit

def estimate_memory_usage(config):
    """Estimate memory usage for given configuration."""
    vol_size = np.prod(config['volume'])
    det_size = np.prod(config['detector'])
    views = config['views']
    
    # Rough estimation in GB
    volume_memory = vol_size * 4 / 1e9  # float32
    sinogram_memory = views * det_size * 4 / 1e9
    
    return volume_memory + sinogram_memory * 2  # Factor for intermediate results
```

## Advanced 3D Reconstruction Techniques

### Regularized Iterative Reconstruction

```python
class RegularizedConeModel(nn.Module):
    """Cone beam model with regularization."""
    def __init__(self, volume_shape, angles, det_u, det_v, du, dv, 
                 source_distance, isocenter_distance, reg_weight=0.01):
        super().__init__()
        self.reco = nn.Parameter(torch.zeros(volume_shape))
        self.angles = angles
        self.det_u = det_u
        self.det_v = det_v
        self.du = du
        self.dv = dv
        self.source_distance = source_distance
        self.isocenter_distance = isocenter_distance
        self.reg_weight = reg_weight

    def forward(self, x):
        updated_reco = x + self.reco
        current_sino = ConeProjectorFunction.apply(
            updated_reco, self.angles, 
            self.det_u, self.det_v, self.du, self.dv, 
            self.source_distance, self.isocenter_distance)
        
        # Add total variation regularization
        tv_loss = self.total_variation_3d(updated_reco)
        
        return current_sino, updated_reco, tv_loss

    def total_variation_3d(self, volume):
        """Compute 3D total variation regularization."""
        diff_z = torch.abs(volume[1:, :, :] - volume[:-1, :, :])
        diff_y = torch.abs(volume[:, 1:, :] - volume[:, :-1, :])
        diff_x = torch.abs(volume[:, :, 1:] - volume[:, :, :-1])
        
        return self.reg_weight * (diff_z.mean() + diff_y.mean() + diff_x.mean())
```

### Multi-Resolution Reconstruction

```python
def multi_resolution_reconstruction(phantom, angles, geometry_params, device):
    """Perform multi-resolution reconstruction for better convergence."""
    
    # Start with low resolution
    low_res_phantom = torch.nn.functional.interpolate(
        phantom.unsqueeze(0).unsqueeze(0), 
        size=(32, 32, 32), mode='trilinear').squeeze()
    
    # Reconstruct at low resolution
    low_res_result = reconstruct_at_resolution(
        low_res_phantom, angles, geometry_params, device, epochs=500)
    
    # Upscale and refine
    mid_res_init = torch.nn.functional.interpolate(
        low_res_result.unsqueeze(0).unsqueeze(0), 
        size=(64, 64, 64), mode='trilinear').squeeze()
    
    mid_res_result = reconstruct_at_resolution(
        mid_res_init, angles, geometry_params, device, epochs=300)
    
    # Final high resolution
    high_res_init = torch.nn.functional.interpolate(
        mid_res_result.unsqueeze(0).unsqueeze(0), 
        size=(128, 128, 128), mode='trilinear').squeeze()
    
    final_result = reconstruct_at_resolution(
        high_res_init, angles, geometry_params, device, epochs=200)
    
    return final_result
```

## Common Issues and Solutions

### 3D-Specific Artifacts

#### Cone Beam Artifacts
- **Cause**: Large cone angles, insufficient axial coverage
- **Solution**: Reduce cone angle, increase detector size in v-direction
- **Detection**: Look for streaking artifacts in axial slices

#### Truncation Artifacts
- **Cause**: Object extends beyond detector field of view
- **Solution**: Increase detector size or reduce object size
- **Check**: Verify object boundaries are visible in all projections

#### Ring Artifacts
- **Cause**: Detector calibration errors, bad pixels
- **Solution**: Detector flat-field correction, outlier detection
- **Prevention**: Proper detector calibration procedures

### Memory and Performance Issues

#### Out of Memory Errors
```python
# Solution strategies
try:
    result = large_reconstruction()
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        # Try with reduced parameters
        result = reduced_reconstruction()
```

#### Slow Convergence
```python
# Adaptive learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50)

for epoch in range(epochs):
    loss = train_step()
    scheduler.step(loss)
```

### Quality Assessment

```python
def assess_3d_reconstruction_quality(original, reconstruction):
    """Assess 3D reconstruction quality metrics."""
    
    # Mean Squared Error
    mse = torch.mean((original - reconstruction)**2)
    
    # Peak Signal-to-Noise Ratio
    max_val = torch.max(original)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    # Structural Similarity Index (simplified)
    ssim = compute_3d_ssim(original, reconstruction)
    
    print(f"MSE: {mse.item():.6f}")
    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    return {'mse': mse.item(), 'psnr': psnr.item(), 'ssim': ssim}
```

## Next Steps and Advanced Topics

### Research Directions
- **AI-Enhanced Reconstruction**: Deep learning priors and neural networks
- **Multi-Energy CT**: Spectral imaging and material decomposition
- **Dynamic CT**: 4D reconstruction with temporal regularization
- **Sparse-View CT**: Compressed sensing and iterative reconstruction

### Integration with Other Tools
- **Medical Imaging**: DICOM integration, clinical workflows
- **Industrial CT**: Non-destructive testing applications
- **Micro-CT**: High-resolution imaging techniques

### Performance Scaling
- **Multi-GPU**: Distributed reconstruction across multiple GPUs
- **Cloud Computing**: Scalable reconstruction in cloud environments
- **Real-Time**: Optimization for real-time reconstruction applications

## References

- Feldkamp, L. A., Davis, L. C., & Kress, J. W. (1984). Practical cone-beam algorithm
- Kak, A. C., & Slaney, M. (1988). Principles of computerized tomographic imaging
- Buzug, T. M. (2008). Computed tomography: from photon statistics to modern cone-beam CT
- Wang, G., et al. (2008). A general cone-beam reconstruction algorithm