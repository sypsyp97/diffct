import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffct.differentiable import FanProjectorFunction, FanBackprojectorFunction

def shepp_logan_2d(Nx, Ny):
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

def example_fan_pipeline():
    Nx, Ny = 256, 256
    phantom = shepp_logan_2d(Nx, Ny)
    num_angles = 360
    angles_np = np.linspace(0, 2*math.pi, num_angles, endpoint=False).astype(np.float32)

    num_detectors = 600
    detector_spacing = 1.0
    step_size = 1.0
    source_distance = 800.0
    isocenter_distance = 500.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_torch = torch.tensor(phantom, device=device, requires_grad=True)
    angles_torch = torch.tensor(angles_np, device=device)

    sinogram = FanProjectorFunction.apply(image_torch, angles_torch, num_detectors,
                                          detector_spacing, step_size, source_distance, isocenter_distance)

    sinogram_filt = ramp_filter(sinogram).detach().requires_grad_(True).contiguous()

    reconstruction = FanBackprojectorFunction.apply(sinogram_filt, angles_torch,
                                                    detector_spacing, step_size, Nx, Ny,
                                                    source_distance, isocenter_distance)
    
    reconstruction = reconstruction / num_angles # Normalize by number of angles

    loss = torch.mean((reconstruction - image_torch)**2)
    loss.backward()

    print("Loss:", loss.item())
    print("Center pixel gradient:", image_torch.grad[Nx//2, Ny//2].item())

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
    example_fan_pipeline()