import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from diffct.differentiable import FanProjectorFunction
from diffct.geometry import (sinusoidal_trajectory_2d_fan, custom_trajectory_2d_fan)


def custom_ellipse_trajectory(angles, sid):
    """Custom trajectory: Elliptical path in 2D space."""
    src_pos = torch.zeros((len(angles), 2), device=angles.device, dtype=angles.dtype)
    src_pos[:, 0] = -sid * 1.5 * torch.sin(angles)  # Wider in x-direction
    src_pos[:, 1] = sid * torch.cos(angles)  # Standard in y-direction
    return src_pos


def shepp_logan_2d(shape):
    """Generate 2D Shepp-Logan phantom."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    xx = (xx - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    yy = (yy - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)

    el_params = np.array([
        [0, 0, 0.69, 0.92, 0, 1],
        [0, -0.0184, 0.6624, 0.874, 0, -0.8],
        [0.22, 0, 0.11, 0.31, -np.pi/10.0, -0.2],
        [-0.22, 0, 0.16, 0.41, np.pi/10.0, -0.2],
        [0, 0.35, 0.21, 0.25, 0, 0.1],
        [0, 0.1, 0.046, 0.046, 0, 0.1],
        [0, -0.1, 0.046, 0.046, 0, 0.1],
        [-0.08, -0.605, 0.046, 0.023, 0, 0.1],
        [0, -0.605, 0.023, 0.023, 0, 0.1],
        [0.06, -0.605, 0.023, 0.046, 0, 0.1],
    ], dtype=np.float32)

    # Extract parameters
    x_pos = el_params[:, 0][:, None, None]
    y_pos = el_params[:, 1][:, None, None]
    a_axis = el_params[:, 2][:, None, None]
    b_axis = el_params[:, 3][:, None, None]
    phi = el_params[:, 4][:, None, None]
    val = el_params[:, 5][:, None, None]

    # Broadcast grid
    xc = xx[None, ...] - x_pos
    yc = yy[None, ...] - y_pos

    c = np.cos(phi)
    s = np.sin(phi)

    # Rotation
    xp = c * xc - s * yc
    yp = s * xc + c * yc

    mask = ((xp ** 2) / (a_axis ** 2) + (yp ** 2) / (b_axis ** 2)) <= 1.0

    shepp_logan = np.sum(mask * val, axis=0)
    shepp_logan = np.clip(shepp_logan, 0, 1)
    return shepp_logan


class IterativeRecoModel(nn.Module):
    def __init__(self, image_shape, src_pos, det_center, det_u_vec, n_det, det_spacing, voxel_spacing):
        super().__init__()
        self.reco = nn.Parameter(torch.zeros(image_shape))
        self.src_pos = src_pos
        self.det_center = det_center
        self.det_u_vec = det_u_vec
        self.n_det = n_det
        self.det_spacing = det_spacing
        self.relu = nn.ReLU()  # non-negative constraint
        self.voxel_spacing = voxel_spacing

    def forward(self, x):
        updated_reco = x + self.reco
        current_sino = FanProjectorFunction.apply(updated_reco,
                                                  self.src_pos, self.det_center, self.det_u_vec,
                                                  self.n_det, self.det_spacing, self.voxel_spacing)
        return current_sino, self.relu(updated_reco)


class Pipeline:
    def __init__(self, lr, image_shape, src_pos, det_center, det_u_vec,
                 n_det, det_spacing, voxel_spacing,
                 device, epoches=1000):
        self.epoches = epoches
        self.model = IterativeRecoModel(image_shape, src_pos, det_center, det_u_vec,
                                       n_det, det_spacing, voxel_spacing).to(device)
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


def run_reconstruction(trajectory_name, src_pos, det_center, det_u_vec,
                      phantom_torch, n_det, det_spacing, voxel_spacing, device, epoches=500):
    """Run iterative reconstruction for a given trajectory."""
    print(f"\n{'='*60}")
    print(f"Processing {trajectory_name} Trajectory")
    print(f"{'='*60}")

    Ny, Nx = phantom_torch.shape

    print(f"Source positions range: x=[{src_pos[:, 0].min():.2f}, {src_pos[:, 0].max():.2f}], "
          f"y=[{src_pos[:, 1].min():.2f}, {src_pos[:, 1].max():.2f}]")

    # Generate sinogram
    print("Generating sinogram...")
    real_sinogram = FanProjectorFunction.apply(phantom_torch, src_pos, det_center, det_u_vec,
                                              n_det, det_spacing, voxel_spacing)

    # Iterative reconstruction
    print("Starting iterative reconstruction...")
    pipeline_instance = Pipeline(lr=1e-1,
                                image_shape=(Ny, Nx),
                                src_pos=src_pos,
                                det_center=det_center,
                                det_u_vec=det_u_vec,
                                n_det=n_det,
                                det_spacing=det_spacing,
                                voxel_spacing=voxel_spacing,
                                device=device,
                                epoches=epoches)

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
    Nx, Ny = 128, 128
    phantom_cpu = shepp_logan_2d((Ny, Nx))

    num_views = 360
    n_det = 256
    det_spacing = 1.0
    voxel_spacing = 1.0
    sdd = 600.0
    sid = 400.0
    epoches = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_cpu, device=device, dtype=torch.float32).contiguous()

    # Dictionary to store results
    results = {}

    # 1. Sinusoidal Trajectory
    print("\n" + "="*60)
    print("Generating Sinusoidal Trajectory...")
    print("="*60)
    src_pos, det_center, det_u_vec = sinusoidal_trajectory_2d_fan(
        n_views=num_views, sid=sid, sdd=sdd,
        amplitude=50.0, frequency=3.0, device=device
    )
    loss_values, reco = run_reconstruction(
        "Sinusoidal", src_pos, det_center, det_u_vec,
        phantom_torch, n_det, det_spacing, voxel_spacing, device, epoches
    )
    results['Sinusoidal'] = (loss_values, reco)

    # 2. Elliptical Trajectory (Custom)
    print("\n" + "="*60)
    print("Generating Elliptical (Custom) Trajectory...")
    print("="*60)
    src_pos, det_center, det_u_vec = custom_trajectory_2d_fan(
        n_views=num_views, sid=sid, sdd=sdd,
        source_path_fn=custom_ellipse_trajectory, device=device
    )
    loss_values, reco = run_reconstruction(
        "Elliptical", src_pos, det_center, det_u_vec,
        phantom_torch, n_det, det_spacing, voxel_spacing, device, epoches
    )
    results['Elliptical'] = (loss_values, reco)

    # Plot results
    fig = plt.figure(figsize=(15, 10))

    trajectory_names = ['Sinusoidal', 'Elliptical']

    # Row 1: Loss curves
    for idx, name in enumerate(trajectory_names):
        plt.subplot(3, 2, idx + 1)
        loss_values, _ = results[name]
        plt.plot(loss_values)
        plt.title(f"{name} - Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.grid(True)

    # Row 2: Original phantom
    for idx in range(2):
        plt.subplot(3, 2, idx + 3)
        plt.imshow(phantom_cpu, cmap="gray", vmin=0, vmax=1)
        if idx == 0:
            plt.title("Original Phantom")
        plt.axis("off")
        if idx == 1:
            plt.colorbar()

    # Row 3: Reconstructions
    for idx, name in enumerate(trajectory_names):
        plt.subplot(3, 2, idx + 5)
        _, reco = results[name]
        plt.imshow(reco, cmap="gray", vmin=0, vmax=1)

        # Compute metrics for title
        mse = np.mean((reco - phantom_cpu) ** 2)
        psnr = 10 * np.log10(1.0 / mse)
        plt.title(f"{name}\nPSNR: {psnr:.2f} dB")
        plt.axis("off")
        if idx == 1:
            plt.colorbar()

    plt.tight_layout()
    plt.savefig("fan_trajectory_reconstruction.png", dpi=300, bbox_inches='tight')
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
