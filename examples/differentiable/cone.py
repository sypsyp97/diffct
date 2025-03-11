import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffct.differentiable import ConeProjectorFunction, ConeBackprojectorFunction

def shepp_logan_3d(shape):
    shepp_logan = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
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
    ])
    for i in range(el_params.shape[0]):
        x_pos = el_params[i][0]
        y_pos = el_params[i][1]
        z_pos = el_params[i][2]
        a_axis = el_params[i][3]
        b_axis = el_params[i][4]
        c_axis = el_params[i][5]
        phi = el_params[i][6]
        val = el_params[i][9]
        xc = xx - x_pos
        yc = yy - y_pos
        zc = zz - z_pos
        c = np.cos(phi)
        s = np.sin(phi)
        Rz_phi = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])
        xp = xc*Rz_phi[0,0] + yc*Rz_phi[0,1] + zc*Rz_phi[0,2]
        yp = xc*Rz_phi[1,0] + yc*Rz_phi[1,1] + zc*Rz_phi[1,2]
        zp = xc*Rz_phi[2,0] + yc*Rz_phi[2,1] + zc*Rz_phi[2,2]
        mask = (xp**2)/(a_axis*a_axis) + (yp**2)/(b_axis*b_axis) + (zp**2)/(c_axis*c_axis) <= 1.0
        shepp_logan[mask] += val
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

def example_cone_pipeline():
    Nx, Ny, Nz = 128, 128, 128
    phantom_cpu = shepp_logan_3d((Nx, Ny, Nz))

    num_views = 360
    angles_np = np.linspace(0, 2*math.pi, num_views, endpoint=False).astype(np.float32)

    det_u, det_v = 256, 256
    du, dv = 1.0, 1.0
    step_size = 1.0
    source_distance = 600.0
    isocenter_distance = 400.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device, requires_grad=True)
    angles_torch = torch.tensor(angles_np, device=device)

    sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch, Nx, Ny, Nz,
                                           det_u, det_v, du, dv, step_size,
                                           source_distance, isocenter_distance)

    sinogram_filt = ramp_filter_3d(sinogram).detach().requires_grad_(True).contiguous()

    reconstruction = ConeBackprojectorFunction.apply(sinogram_filt, angles_torch,
                                                     Nx, Ny, Nz, det_u, det_v, du, dv,
                                                     step_size, source_distance, isocenter_distance)
    
    reconstruction = reconstruction / num_views # Normalize

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

if __name__ == "__main__":
    example_cone_pipeline()