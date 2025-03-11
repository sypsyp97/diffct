import numpy as np
import torch
import matplotlib.pyplot as plt
from cone_cuda import forward_cone_3d, back_cone_3d

def shepp_logan_3d(shape):
    shepp_logan = np.zeros(shape, dtype=np.float32)
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
    ])

    for i in range(el_params.shape[0]):
        x_pos = el_params[i][0]
        y_pos = el_params[i][1]
        z_pos = el_params[i][2]
        a_axis = el_params[i][3]
        b_axis = el_params[i][4]
        c_axis = el_params[i][5]
        phi = el_params[i][6]
        val = el_params[i][9]
        xc = xx - x_pos
        yc = yy - y_pos
        zc = zz - z_pos
        c = np.cos(phi)
        s = np.sin(phi)
        Rz_phi = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])
        xp = xc*Rz_phi[0,0] + yc*Rz_phi[0,1] + zc*Rz_phi[0,2]
        yp = xc*Rz_phi[1,0] + yc*Rz_phi[1,1] + zc*Rz_phi[1,2]
        zp = xc*Rz_phi[2,0] + yc*Rz_phi[2,1] + zc*Rz_phi[2,2]
        mask = (
            (xp**2)/(a_axis*a_axis)
            + (yp**2)/(b_axis*b_axis)
            + (zp**2)/(c_axis*c_axis)
            <= 1.0
        )
        shepp_logan[mask] += val

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
    Nx, Ny, Nz = 256, 256, 256
    phantom = shepp_logan_3d((Nx, Ny, Nz))
    num_views = 180
    num_det_u = 512
    num_det_v = 512
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

    # Uncomment to normalize the reconstruction to be in [0, 1]

    # mxp = reco.max()
    # if mxp > 1e-12:
    #     reco /= mxp
    # reco = np.clip(reco, 0, 1)

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

if __name__ == "__main__":
    main()
