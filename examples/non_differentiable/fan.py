import numpy as np
import torch
import matplotlib.pyplot as plt
from diffct.non_differentiable import forward_fan_2d, back_fan_2d

def shepp_logan_2d(Nx, Ny):
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
                if xprime**2/(a*a) + yprime**2/(b*b) <= 1.0:
                    val += ampl
            phantom[ix, iy] = val
    phantom = np.clip(phantom, 0.0, 1.0)
    return phantom

def ramp_filter(sinogram_tensor):
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

def main():
    Nx, Ny = 256, 256
    phantom = shepp_logan_2d(Nx, Ny)
    num_views = 360
    num_detectors = 600
    detector_spacing = 1.0
    angles = np.linspace(0, 2*np.pi, num_views, endpoint=False).astype(np.float64)
    source_distance = 800.0
    isocenter_distance = 500.0

    sinogram_np = forward_fan_2d(
        phantom, num_views, num_detectors, detector_spacing,
        angles, source_distance, isocenter_distance
    )

    sinogram_torch = torch.from_numpy(sinogram_np)

    # --- FBP weighting and filtering ---
    # For fan-beam FBP, projections must be weighted before filtering.
    # Weight = cos(gamma), where gamma is the fan angle for each detector.
    # Detector coordinates
    u = (torch.arange(num_detectors, dtype=sinogram_torch.dtype, device=sinogram_torch.device) - (num_detectors - 1) / 2) * detector_spacing
    # Fan angles
    gamma = torch.atan(u / source_distance)
    # Cosine weights
    weights = torch.cos(gamma).unsqueeze(0)  # Shape (1, num_detectors) for broadcasting
    
    # Apply weights and then filter
    sino_weighted = sinogram_torch * weights
    sino_filt = ramp_filter(sino_weighted).contiguous().numpy()

    reco = back_fan_2d(
        sino_filt, Nx, Ny, detector_spacing,
        angles, source_distance, isocenter_distance
    )

    # --- FBP normalization ---
    # The backprojection is a sum over all angles. To approximate the integral,
    # we need to multiply by the angular step d_beta.
    # The fan-beam FBP formula also includes a factor of 1/2 when integrating over [0, 2*pi].
    # d_beta = 2 * pi / num_views
    # Normalization factor = (1/2) * d_beta = pi / num_views
    reco = reco * (np.pi / num_views)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(phantom, cmap='gray')
    plt.axis('off')
    plt.title("Phantom")

    plt.subplot(1, 3, 2)
    plt.imshow(sinogram_np, cmap='gray', aspect='auto')
    plt.axis('off')
    plt.title("Fan Beam Sinogram")

    plt.subplot(1, 3, 3)
    plt.imshow(reco, cmap='gray')
    plt.axis('off')
    plt.title("Fan Beam Reconstruction")
    plt.tight_layout()
    plt.show()

    # print data range of the phantom and reco
    print("Phantom data range:", phantom.min(), phantom.max())
    print("Reconstruction data range:", reco.min(), reco.max())

if __name__ == "__main__":
    main()