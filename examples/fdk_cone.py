import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffct.differentiable import ConeProjectorFunction, ConeBackprojectorFunction


def shepp_logan_3d(shape):
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    xx = (xx - (shape[2] - 1) / 2) / ((shape[2] - 1) / 2)
    yy = (yy - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    zz = (zz - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)
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

    # Extract parameters for vectorization
    x_pos = el_params[:, 0][:, None, None, None]
    y_pos = el_params[:, 1][:, None, None, None]
    z_pos = el_params[:, 2][:, None, None, None]
    a_axis = el_params[:, 3][:, None, None, None]
    b_axis = el_params[:, 4][:, None, None, None]
    c_axis = el_params[:, 5][:, None, None, None]
    phi = el_params[:, 6][:, None, None, None]
    val = el_params[:, 9][:, None, None, None]

    # Broadcast grid to ellipsoid axis
    xc = xx[None, ...] - x_pos
    yc = yy[None, ...] - y_pos
    zc = zz[None, ...] - z_pos

    c = np.cos(phi)
    s = np.sin(phi)

    # Only rotation around z, so can vectorize:
    xp = c * xc - s * yc
    yp = s * xc + c * yc
    zp = zc

    mask = (
        (xp ** 2) / (a_axis ** 2)
        + (yp ** 2) / (b_axis ** 2)
        + (zp ** 2) / (c_axis ** 2)
        <= 1.0
    )

    # Use broadcasting to sum all ellipsoid contributions
    shepp_logan = np.sum(mask * val, axis=0)
    shepp_logan = np.clip(shepp_logan, 0, 1)
    return shepp_logan

def ramp_filter_3d(sinogram_tensor):
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

def main():
    Nx, Ny, Nz = 128, 128, 128
    phantom_cpu = shepp_logan_3d((Nx, Ny, Nz))

    num_views = 360
    angles_np = np.linspace(0, 2*math.pi, num_views, endpoint=False).astype(np.float32)

    det_u, det_v = 256, 256
    du, dv = 1.0, 1.0
    source_distance = 900.0
    isocenter_distance = 600.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device, requires_grad=True)
    angles_torch = torch.tensor(angles_np, device=device)

    sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch,
                                           det_u, det_v, du, dv,
                                           source_distance, isocenter_distance)

    # --- FDK weighting and filtering ---
    # For FDK, projections must be weighted before filtering.
    # Weight = D / sqrt(D^2 + u^2 + v^2), where D is source_distance
    # and (u,v) are detector coordinates.
    u_coords = (torch.arange(det_u, dtype=phantom_torch.dtype, device=device) - (det_u - 1) / 2) * du
    v_coords = (torch.arange(det_v, dtype=phantom_torch.dtype, device=device) - (det_v - 1) / 2) * dv

    # Reshape for broadcasting over sinogram of shape (views, u, v)
    u_coords = u_coords.view(1, det_u, 1)
    v_coords = v_coords.view(1, 1, det_v)
    
    weights = source_distance / torch.sqrt(source_distance**2 + u_coords**2 + v_coords**2)
    
    # Apply weights and then filter
    sino_weighted = sinogram * weights
    sinogram_filt = ramp_filter_3d(sino_weighted).detach().requires_grad_(True).contiguous()

    reconstruction = ConeBackprojectorFunction.apply(sinogram_filt, angles_torch, Nx, Ny, Nz,
                                                    du, dv, source_distance, isocenter_distance)
    
    # --- FDK normalization ---
    # The backprojection is a sum over all angles. To approximate the integral,
    # we need to multiply by the angular step d_beta.
    # The FDK formula also includes a factor of 1/2 when integrating over [0, 2*pi].
    # d_beta = 2 * pi / num_views
    # Normalization factor = (1/2) * d_beta = pi / num_views
    reconstruction = reconstruction * (math.pi / num_views)

    loss = torch.mean((reconstruction - phantom_torch)**2)
    loss.backward()

    print("Cone Beam Example with user-defined geometry:")
    print("Loss:", loss.item())
    print("Volume center voxel gradient:", phantom_torch.grad[Nx//2, Ny//2, Nz//2].item())
    print("Reconstruction shape:", reconstruction.shape)

    reconstruction_cpu = reconstruction.detach().cpu().numpy()
    sinogram_cpu = sinogram.detach().cpu().numpy()
    mid_slice = Nz // 2

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(phantom_cpu[:,:,mid_slice], cmap='gray')
    plt.title("Phantom mid-slice")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(sinogram_cpu[num_views//2], cmap='gray')
    plt.title("Sinogram mid-view")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(reconstruction_cpu[:,:,mid_slice], cmap='gray')
    plt.title("Recon mid-slice")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # print data range of the phantom and reco
    print("Phantom data range:", phantom_cpu.min(), phantom_cpu.max())
    print("Reco data range:", reconstruction_cpu.min(), reconstruction_cpu.max())

if __name__ == "__main__":
    main()