import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from diffct.differentiable import (
    ConeProjectorFunction,
    angular_integration_weights,
    cone_cosine_weights,
    cone_weighted_backproject,
    ramp_filter_1d,
)


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

def main():
    Nx, Ny, Nz = 128, 128, 128
    phantom_cpu = shepp_logan_3d((Nz, Ny, Nx))

    num_views = 360
    angles_np = np.linspace(0, 2*math.pi, num_views, endpoint=False).astype(np.float32)

    det_u, det_v = 256, 256
    du, dv = 1.0, 1.0
    detector_offset_u = 0.0
    detector_offset_v = 0.0
    sdd = 900.0
    sid = 600.0

    voxel_spacing = 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device, dtype=torch.float32).contiguous()
    angles_torch = torch.tensor(angles_np, device=device, dtype=torch.float32)

    sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch,
                                           det_u, det_v, du, dv,
                                           sdd, sid, voxel_spacing)

    # --- FDK weighting and filtering ---
    # 1) FDK cosine pre-weighting
    weights = cone_cosine_weights(
        det_u,
        det_v,
        du,
        dv,
        sdd,
        detector_offset_u=detector_offset_u,
        detector_offset_v=detector_offset_v,
        device=device,
        dtype=phantom_torch.dtype,
    ).unsqueeze(0)
    sino_weighted = sinogram * weights

    # 2) Ramp filter along detector-u rows
    sinogram_filt = ramp_filter_1d(sino_weighted, dim=1).contiguous()

    # 3) Angle-integration weights
    d_beta = angular_integration_weights(angles_torch, redundant_full_scan=True).view(-1, 1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # 4) Weighted cone-beam backprojection
    reconstruction = F.relu(
        cone_weighted_backproject(
            sinogram_filt,
            angles_torch,
            Nz,
            Ny,
            Nx,
            du,
            dv,
            sdd,
            sid,
            voxel_spacing=voxel_spacing,
            detector_offset_u=detector_offset_u,
            detector_offset_v=detector_offset_v,
        )
    )

    loss = torch.mean((reconstruction - phantom_torch)**2)

    print("Cone Beam Example with user-defined geometry:")
    print("Loss:", loss.item())
    print("Reconstruction shape:", reconstruction.shape)

    reconstruction_cpu = reconstruction.detach().cpu().numpy()
    sinogram_cpu = sinogram.detach().cpu().numpy()
    mid_slice = Nz // 2

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(phantom_cpu[mid_slice, :,:], cmap='gray')
    plt.title("Phantom mid-slice")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(sinogram_cpu[num_views//2].T, cmap='gray', origin='lower') # Transpose for correct orientation
    plt.title("Sinogram mid-view")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(reconstruction_cpu[mid_slice, :, :], cmap='gray')
    plt.title("Recon mid-slice")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # print data range of the phantom and reco
    print("Phantom data range:", phantom_cpu.min(), phantom_cpu.max())
    print("Reco data range:", reconstruction_cpu.min(), reconstruction_cpu.max())

if __name__ == "__main__":
    main()
