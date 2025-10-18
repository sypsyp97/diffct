import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import ConeProjectorFunction
from diffct.geometry import (spiral_trajectory_3d, sinusoidal_trajectory_3d,
                             saddle_trajectory_3d, custom_trajectory_3d)

def custom_figure8_trajectory(angles, sid):
    """Custom trajectory: Figure-8 pattern in 3D space."""
    src_pos = torch.zeros((len(angles), 3), device=angles.device, dtype=angles.dtype)
    src_pos[:, 0] = -sid * torch.sin(angles)
    src_pos[:, 1] = sid * torch.cos(angles) * torch.sin(angles)  # Figure-8 in y
    src_pos[:, 2] = 50 * torch.sin(2 * angles)  # Z oscillation
    return src_pos


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

def run_reconstruction(trajectory_name, src_pos, det_center, det_u_vec, det_v_vec,
                      phantom_torch, det_u, det_v, du, dv, voxel_spacing, device, epoches=1000):
    """Run iterative reconstruction for a given trajectory."""
    print(f"\n{'='*60}")
    print(f"Processing {trajectory_name} Trajectory")
    print(f"{'='*60}")

    Nz, Ny, Nx = phantom_torch.shape

    print(f"Source positions range: x=[{src_pos[:, 0].min():.2f}, {src_pos[:, 0].max():.2f}], "
          f"y=[{src_pos[:, 1].min():.2f}, {src_pos[:, 1].max():.2f}], "
          f"z=[{src_pos[:, 2].min():.2f}, {src_pos[:, 2].max():.2f}]")

    # Generate sinogram
    print("Generating sinogram...")
    real_sinogram = ConeProjectorFunction.apply(phantom_torch, src_pos, det_center,
                                               det_u_vec, det_v_vec,
                                               det_u, det_v, du, dv, voxel_spacing)

    # Iterative reconstruction
    print("Starting iterative reconstruction...")
    pipeline_instance = Pipeline(lr=1e-1,
                                 volume_shape=(Nz, Ny, Nx),
                                 src_pos=src_pos,
                                 det_center=det_center,
                                 det_u_vec=det_u_vec,
                                 det_v_vec=det_v_vec,
                                 det_u=det_u, det_v=det_v,
                                 du=du, dv=dv, voxel_spacing=voxel_spacing,
                                 device=device, epoches=epoches)

    ini_guess = torch.zeros_like(phantom_torch)
    loss_values, trained_model = pipeline_instance.train(ini_guess, real_sinogram)
    reco = trained_model(ini_guess)[1].squeeze().cpu().detach().numpy()

    # Compute metrics
    phantom_cpu = phantom_torch.cpu().numpy()
    mse = np.mean((reco - phantom_cpu) ** 2)
    psnr = 10 * np.log10(1.0 / mse)

    print(f"\nReconstruction Metrics for {trajectory_name}:")
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")

    return loss_values, reco


def main():
    Nx, Ny, Nz = 64, 64, 64
    phantom_cpu = shepp_logan_3d((Nz, Ny, Nx))

    num_views = 180
    det_u, det_v = 128, 128
    du, dv = 1.0, 1.0
    voxel_spacing = 1.0
    sdd = 600.0
    sid = 400.0
    epoches = 500  # Reduced for faster demonstration

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device, dtype=torch.float32).contiguous()

    # Dictionary to store results
    results = {}

    # 1. Spiral Trajectory
    print("\n" + "="*60)
    print("Generating Spiral Trajectory...")
    print("="*60)
    src_pos, det_center, det_u_vec, det_v_vec = spiral_trajectory_3d(
        n_views=num_views, sid=sid, sdd=sdd,
        z_range=80.0, n_turns=2.0, device=device
    )
    loss_values, reco = run_reconstruction(
        "Spiral", src_pos, det_center, det_u_vec, det_v_vec,
        phantom_torch, det_u, det_v, du, dv, voxel_spacing, device, epoches
    )
    results['Spiral'] = (loss_values, reco)

    # 2. Sinusoidal Trajectory
    print("\n" + "="*60)
    print("Generating Sinusoidal Trajectory...")
    print("="*60)
    src_pos, det_center, det_u_vec, det_v_vec = sinusoidal_trajectory_3d(
        n_views=num_views, sid=sid, sdd=sdd,
        amplitude=50.0, frequency=3.0, device=device
    )
    loss_values, reco = run_reconstruction(
        "Sinusoidal", src_pos, det_center, det_u_vec, det_v_vec,
        phantom_torch, det_u, det_v, du, dv, voxel_spacing, device, epoches
    )
    results['Sinusoidal'] = (loss_values, reco)

    # 3. Saddle Trajectory
    print("\n" + "="*60)
    print("Generating Saddle Trajectory...")
    print("="*60)
    src_pos, det_center, det_u_vec, det_v_vec = saddle_trajectory_3d(
        n_views=num_views, sid=sid, sdd=sdd,
        z_amplitude=60.0, radial_amplitude=40.0, device=device
    )
    loss_values, reco = run_reconstruction(
        "Saddle", src_pos, det_center, det_u_vec, det_v_vec,
        phantom_torch, det_u, det_v, du, dv, voxel_spacing, device, epoches
    )
    results['Saddle'] = (loss_values, reco)

    # 4. Custom Trajectory (Figure-8)
    print("\n" + "="*60)
    print("Generating Custom (Figure-8) Trajectory...")
    print("="*60)
    src_pos, det_center, det_u_vec, det_v_vec = custom_trajectory_3d(
        n_views=num_views, sid=sid, sdd=sdd,
        source_path_fn=custom_figure8_trajectory, device=device
    )
    loss_values, reco = run_reconstruction(
        "Custom (Figure-8)", src_pos, det_center, det_u_vec, det_v_vec,
        phantom_torch, det_u, det_v, du, dv, voxel_spacing, device, epoches
    )
    results['Custom'] = (loss_values, reco)

    # Plot results - Combined comparison
    fig = plt.figure(figsize=(20, 12))
    mid_slice = Nz // 2

    trajectory_names = ['Spiral', 'Sinusoidal', 'Saddle', 'Custom']

    # Row 1: Loss curves
    for idx, name in enumerate(trajectory_names):
        plt.subplot(3, 4, idx + 1)
        loss_values, _ = results[name]
        plt.plot(loss_values)
        plt.title(f"{name} - Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.grid(True)

    # Row 2: Original phantom (same for all)
    for idx in range(4):
        plt.subplot(3, 4, idx + 5)
        plt.imshow(phantom_cpu[mid_slice, :, :], cmap="gray", vmin=0, vmax=1)
        if idx == 0:
            plt.title("Original Phantom")
        plt.axis("off")
        if idx == 3:
            plt.colorbar()

    # Row 3: Reconstructions
    for idx, name in enumerate(trajectory_names):
        plt.subplot(3, 4, idx + 9)
        _, reco = results[name]
        plt.imshow(reco[mid_slice, :, :], cmap="gray", vmin=0, vmax=1)

        # Compute metrics for title
        mse = np.mean((reco - phantom_cpu) ** 2)
        psnr = 10 * np.log10(1.0 / mse)
        plt.title(f"{name}\nPSNR: {psnr:.2f} dB")
        plt.axis("off")
        if idx == 3:
            plt.colorbar()

    plt.tight_layout()
    plt.savefig("multi_trajectory_reconstruction.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("RECONSTRUCTION SUMMARY")
    print("="*60)
    print(f"{'Trajectory':<20} {'MSE':<12} {'PSNR (dB)':<12}")
    print("-" * 60)
    for name in trajectory_names:
        _, reco = results[name]
        mse = np.mean((reco - phantom_cpu) ** 2)
        psnr = 10 * np.log10(1.0 / mse)
        print(f"{name:<20} {mse:<12.6f} {psnr:<12.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
