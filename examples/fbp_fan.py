import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from diffct.differentiable import (
    FanProjectorFunction,
    angular_integration_weights,
    fan_cosine_weights,
    fan_weighted_backproject,
    parker_weights,
    ramp_filter_1d,
)


def shepp_logan_2d(Nx, Ny):
    Nx = int(Nx)
    Ny = int(Ny)
    phantom = np.zeros((Ny, Nx), dtype=np.float32)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 1.0),
        (0.0, -0.0184, 0.6624, 0.8740, 0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18.0, -0.8),
        (-0.22, 0.0, 0.16, 0.41, 18.0, -0.8),
        (0.0, 0.35, 0.21, 0.25, 0, 0.7),
    ]
    cx = (Nx - 1)*0.5
    cy = (Ny - 1)*0.5
    for ix in range(Nx):
        for iy in range(Ny):
            xnorm = (ix - cx)/(Nx/2)
            ynorm = (iy - cy)/(Ny/2)
            val = 0.0
            for (x0, y0, a, b, angdeg, ampl) in ellipses:
                th = np.deg2rad(angdeg)
                xprime = (xnorm - x0)*np.cos(th) + (ynorm - y0)*np.sin(th)
                yprime = -(xnorm - x0)*np.sin(th) + (ynorm - y0)*np.cos(th)
                if xprime*xprime/(a*a) + yprime*yprime/(b*b) <= 1.0:
                    val += ampl
            phantom[iy, ix] = val
    phantom = np.clip(phantom, 0.0, 1.0)
    return phantom

def main():
    Nx, Ny = 256, 256
    phantom = shepp_logan_2d(Nx, Ny)
    num_angles = 360
    angles_np = np.linspace(0, 2*math.pi, num_angles, endpoint=False).astype(np.float32)

    num_detectors = 600
    detector_spacing = 1.0
    detector_offset = 0.0
    voxel_spacing = 1.0
    sdd = 800.0
    sid = 500.0
    apply_parker = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_torch = torch.tensor(phantom, device=device, dtype=torch.float32)
    angles_torch = torch.tensor(angles_np, device=device, dtype=torch.float32)

    sinogram = FanProjectorFunction.apply(image_torch, angles_torch, num_detectors,
                                          detector_spacing, sdd, sid, voxel_spacing)

    # --- FBP weighting and filtering ---
    # 1) Optional Parker redundancy weighting for short-scan trajectories
    if apply_parker:
        parker = parker_weights(angles_torch, num_detectors, detector_spacing, sdd, detector_offset)
        sinogram = sinogram * parker

    # 2) Fan-beam cosine pre-weighting
    weights = fan_cosine_weights(
        num_detectors,
        detector_spacing,
        sdd,
        detector_offset=detector_offset,
        device=device,
        dtype=image_torch.dtype,
    ).unsqueeze(0)
    sino_weighted = sinogram * weights

    # 3) Ramp filter along detector axis
    sinogram_filt = ramp_filter_1d(sino_weighted, dim=1)

    # 4) Angle-integration weights
    d_beta = angular_integration_weights(angles_torch, redundant_full_scan=(not apply_parker)).view(-1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # 5) Weighted fan-beam backprojection
    reconstruction = F.relu(
        fan_weighted_backproject(
            sinogram_filt,
            angles_torch,
            detector_spacing,
            Ny,
            Nx,
            sdd,
            sid,
            voxel_spacing=voxel_spacing,
            detector_offset=detector_offset,
        )
    )

    loss = torch.mean((reconstruction - image_torch)**2)

    print("Loss:", loss.item())

    sinogram_cpu = sinogram.detach().cpu().numpy()
    reco_cpu = reconstruction.detach().cpu().numpy()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(phantom, cmap='gray')
    plt.title("Phantom")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(sinogram_cpu, cmap='gray', aspect='auto')
    plt.title("Fan Sinogram")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(reco_cpu, cmap='gray')
    plt.title("Fan Reconstruction")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # print data range of the phantom and reco
    print("Phantom range:", phantom.min(), phantom.max())
    print("Reco range:", reco_cpu.min(), reco_cpu.max())

if __name__ == "__main__":
    main()
