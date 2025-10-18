import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import ConeProjectorFunction, random_trajectory_3d

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
    def __init__(self, volume_shape, src_pos, det_center, det_u_vec, det_v_vec, det_u, det_v, du, dv, voxel_spacing):
        super().__init__()
        self.reco = nn.Parameter(torch.zeros(volume_shape))
        self.src_pos = src_pos
        self.det_center = det_center
        self.det_u_vec = det_u_vec
        self.det_v_vec = det_v_vec
        self.det_u = det_u
        self.det_v = det_v
        self.du = du
        self.dv = dv
        self.relu = nn.ReLU() # non negative constraint
        self.voxel_spacing = voxel_spacing

    def forward(self, x):
        updated_reco = x + self.reco
        current_sino = ConeProjectorFunction.apply(updated_reco,
                                                   self.src_pos, self.det_center,
                                                   self.det_u_vec, self.det_v_vec,
                                                   self.det_u, self.det_v,
                                                   self.du, self.dv, self.voxel_spacing)
        return current_sino, self.relu(updated_reco)

class Pipeline:
    def __init__(self, lr, volume_shape, src_pos, det_center, det_u_vec, det_v_vec,
                 det_u, det_v, du, dv, voxel_spacing,
                 device, epoches=1000):

        self.epoches = epoches
        self.model = IterativeRecoModel(volume_shape, src_pos, det_center, det_u_vec, det_v_vec,
                                        det_u, det_v, du, dv, voxel_spacing).to(device)

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
    phantom_cpu = shepp_logan_3d((Nz, Ny, Nx))

    num_views = 180
    det_u, det_v = 128, 128
    du, dv = 1.0, 1.0
    voxel_spacing = 1.0
    sdd = 600.0
    sid = 400.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device, dtype=torch.float32).contiguous()

    # Generate random trajectory with perturbations
    # sid_std: 5% variation in source-to-isocenter distance
    # pos_std: 5mm random position offsets
    # angle_std: 0.05 radians (~3 degrees) angular perturbations
    print("Generating random trajectory with perturbations...")
    src_pos, det_center, det_u_vec, det_v_vec = random_trajectory_3d(
        n_views=num_views, sid_mean=sid, sdd_mean=sdd,
        sid_std=20.0,  # 5% of 400mm
        pos_std=5.0,   # 5mm position noise
        angle_std=0.05,  # ~3 degrees
        seed=42,  # for reproducibility
        device=device
    )

    print(f"Source positions range: x=[{src_pos[:, 0].min():.2f}, {src_pos[:, 0].max():.2f}], "
          f"y=[{src_pos[:, 1].min():.2f}, {src_pos[:, 1].max():.2f}], "
          f"z=[{src_pos[:, 2].min():.2f}, {src_pos[:, 2].max():.2f}]")

    # Generate the "real" sinogram using the random trajectory
    print("Generating sinogram...")
    real_sinogram = ConeProjectorFunction.apply(phantom_torch, src_pos, det_center,
                                               det_u_vec, det_v_vec,
                                               det_u, det_v, du, dv, voxel_spacing)

    print("Starting iterative reconstruction...")
    pipeline_instance = Pipeline(lr=1e-1,
                                 volume_shape=(Nz, Ny, Nx),
                                 src_pos=src_pos,
                                 det_center=det_center,
                                 det_u_vec=det_u_vec,
                                 det_v_vec=det_v_vec,
                                 det_u=det_u, det_v=det_v,
                                 du=du, dv=dv, voxel_spacing=voxel_spacing,
                                 device=device, epoches=1000)

    ini_guess = torch.zeros_like(phantom_torch)

    loss_values, trained_model = pipeline_instance.train(ini_guess, real_sinogram)

    reco = trained_model(ini_guess)[1].squeeze().cpu().detach().numpy()

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(loss_values)
    plt.title("Loss Curve (Random Trajectory)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.grid(True)

    # Original phantom
    mid_slice = Nz // 2
    plt.subplot(1, 3, 2)
    plt.imshow(phantom_cpu[mid_slice, :, :], cmap="gray")
    plt.title("Original Phantom Mid-Slice")
    plt.axis("off")
    plt.colorbar()

    # Reconstruction
    plt.subplot(1, 3, 3)
    plt.imshow(reco[mid_slice, :, :], cmap="gray")
    plt.title("Reconstructed Mid-Slice\n(Random Trajectory)")
    plt.axis("off")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("random_trajectory_reconstruction.png", dpi=150)
    plt.show()

    # Compute reconstruction metrics
    mse = np.mean((reco - phantom_cpu) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    print(f"\nReconstruction Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Phantom range: [{phantom_cpu.min():.3f}, {phantom_cpu.max():.3f}]")
    print(f"Reconstruction range: [{reco.min():.3f}, {reco.max():.3f}]")

if __name__ == "__main__":
    main()
