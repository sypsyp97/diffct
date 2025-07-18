import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import ConeProjectorFunction

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

class IterativeRecoModel(nn.Module):
    def __init__(self, volume_shape, angles, det_u, det_v, du, dv, source_distance, isocenter_distance):
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
        current_sino = ConeProjectorFunction.apply(updated_reco, 
                                                   self.angles, 
                                                   self.det_u, self.det_v, 
                                                   self.du, self.dv, 
                                                   self.source_distance, self.isocenter_distance)
        return current_sino, updated_reco

class Pipeline:
    def __init__(self, lr, volume_shape, angles, 
                 det_u, det_v, du, dv, 
                 source_distance, isocenter_distance, 
                 device, epoches=1000):
        
        self.epoches = epoches
        self.model = IterativeRecoModel(volume_shape, angles,
                                        det_u, det_v, du, dv, 
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

def main():
    Nx, Ny, Nz = 64, 64, 64
    phantom_cpu = shepp_logan_3d((Nx, Ny, Nz))

    num_views = 180
    angles_np = np.linspace(0, 2 * math.pi, num_views, endpoint=False).astype(np.float32)

    det_u, det_v = 128, 128
    du, dv = 1.0, 1.0
    source_distance = 600.0
    isocenter_distance = 400.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device)
    angles_torch = torch.tensor(angles_np, device=device)

    # Generate the "real" sinogram
    real_sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch,
                                               det_u, det_v, du, dv,
                                               source_distance, isocenter_distance)

    pipeline_instance = Pipeline(lr=1e-1, 
                                 volume_shape=(Nz,Ny,Nx), 
                                 angles=angles_torch, 
                                 det_u=det_u, det_v=det_v, 
                                 du=du, dv=dv, 
                                 source_distance=source_distance, 
                                 isocenter_distance=isocenter_distance, 
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

    mid_slice = Nz // 2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(phantom_cpu[mid_slice, :, :], cmap="gray")
    plt.title("Original Phantom Mid-Slice")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reco[mid_slice, :, :], cmap="gray")
    plt.title("Reconstructed Mid-Slice")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()