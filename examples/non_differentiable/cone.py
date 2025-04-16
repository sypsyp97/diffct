import numpy as np
import torch
import matplotlib.pyplot as plt
from diffct.non_differentiable import forward_cone_3d, back_cone_3d

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
    phantom = shepp_logan_3d((Nx, Ny, Nz))
    num_views = 360
    num_det_u = 256
    num_det_v = 256
    du = 1.0
    dv = 1.0
    angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)
    source_distance = 600.0
    isocenter_distance = 400.0
    step_size = 1.0

    sinogram = forward_cone_3d(
        phantom, num_views, num_det_u, num_det_v, du, dv,
        angles, source_distance, isocenter_distance, step_size
    )

    sino = torch.from_numpy(sinogram)
    sino_filt = ramp_filter_3d(sino).contiguous().numpy()

    reco = back_cone_3d(
        sino_filt, Nx, Ny, Nz, du, dv, angles,
        source_distance, isocenter_distance, step_size
    )

    reco = reco / num_views  # Normalize by number of angles

    midz = Nz // 2

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(phantom[:,:,midz], cmap='gray')
    plt.axis('off')
    plt.title("Phantom (Mid-Z)")

    plt.subplot(1,3,2)
    plt.imshow(sinogram[num_views//2], cmap='gray')
    plt.axis('off')
    plt.title("Sinogram (Slice)")

    plt.subplot(1,3,3)
    plt.imshow(reco[:,:,midz], cmap='gray')
    plt.axis('off')
    plt.title("Reconstruction (Mid-Z)")
    plt.tight_layout()
    plt.show()

    # print data range of the phantom and reco
    print("Phantom range:", phantom.min(), phantom.max())
    print("Reco range:", reco.min(), reco.max())

if __name__ == "__main__":
    main()
