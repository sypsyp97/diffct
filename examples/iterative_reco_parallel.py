import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import ParallelProjectorFunction


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

class IterativeRecoModel(nn.Module):
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

def main():
    Nx, Ny = 128, 128
    phantom_cpu = shepp_logan_2d(Nx, Ny)

    num_views = 360
    angles_np = np.linspace(0, 2 * math.pi, num_views, endpoint=False).astype(np.float32)

    num_detectors = 256
    detector_spacing = 0.75

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device)
    angles_torch = torch.tensor(angles_np, device=device)

    # Generate the "real" sinogram
    real_sinogram = ParallelProjectorFunction.apply(phantom_torch, angles_torch,
                                                    num_detectors, detector_spacing)

    pipeline_instance = Pipeline(lr=1e-1,
                                 volume_shape=(Ny, Nx),
                                 angles=angles_torch,
                                 num_detectors=num_detectors,
                                 detector_spacing=detector_spacing,
                                 device=device, epoches=1000)

    ini_guess = torch.zeros_like(phantom_torch)

    loss_values, trained_model = pipeline_instance.train(ini_guess, real_sinogram)

    reco = trained_model(ini_guess)[1].squeeze().cpu().detach().numpy()

    plt.figure()
    plt.plot(loss_values)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(phantom_cpu, cmap="gray")
    plt.title("Original Phantom")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reco, cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()