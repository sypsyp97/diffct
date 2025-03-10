import math
import numpy as np
import torch
from numba import cuda
import matplotlib.pyplot as plt

############################################################################
# Cone Beam: Differentiable Forward Projector
############################################################################

class ConeProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, angles, Nx, Ny, Nz, det_u, det_v, du, dv, step_size, source_distance, isocenter_distance):
        device = volume.device
        num_views = angles.shape[0]

        d_volume = cuda.to_device(volume.detach().cpu().numpy())
        d_angles = cuda.to_device(angles.detach().cpu().numpy())
        d_sino = cuda.device_array((num_views, det_u, det_v), dtype=np.float32)

        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(num_views / threadsperblock[0])
        blockspergrid_y = math.ceil(det_u / threadsperblock[1])
        blockspergrid_z = math.ceil(det_v / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5
        cz = (Nz - 1)*0.5

        ConeProjectorFunction._forward_kernel[blockspergrid, threadsperblock](
            d_volume, Nx, Ny, Nz,
            d_sino, num_views, det_u, det_v,
            du, dv, d_angles,
            step_size,
            source_distance,
            isocenter_distance,
            cx, cy, cz
        )

        sinogram = torch.tensor(d_sino.copy_to_host(), device=device)
        ctx.save_for_backward(volume, angles)
        ctx.intermediate = (Nx, Ny, Nz, det_u, det_v, du, dv, step_size, source_distance, isocenter_distance)
        return sinogram

    @staticmethod
    @cuda.jit
    def _forward_kernel(d_volume, Nx, Ny, Nz,
                        d_sino, num_views, det_u, det_v,
                        du, dv, d_angles,
                        step_size,
                        source_distance,
                        isocenter_distance,
                        cx, cy, cz):
        iview, iu, ivz = cuda.grid(3)
        if iview < num_views and iu < det_u and ivz < det_v:
            angle = d_angles[iview]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (iu - (det_u - 1)*0.5)*du
            v = (ivz - (det_v - 1)*0.5)*dv

            sx = -isocenter_distance*sin_a
            sy =  isocenter_distance*cos_a
            sz =  0.0

            dx = (source_distance - isocenter_distance)*sin_a + u*cos_a
            dy = -(source_distance - isocenter_distance)*cos_a + u*sin_a
            dz = v

            rx = dx - sx
            ry = dy - sy
            rz = dz - sz
            length = math.sqrt(rx*rx + ry*ry + rz*rz)
            rx /= length
            ry /= length
            rz /= length

            total_val = 0.0
            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t*rx
                y = sy + t*ry
                z = sz + t*rz

                ix = x + cx
                iy = y + cy
                iz = z + cz
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))
                iz0 = int(math.floor(iz))

                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1 and
                    iz0 >= 0 and iz0 < Nz - 1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    dz2 = iz - iz0
                    c000 = d_volume[ix0,   iy0,   iz0]
                    c100 = d_volume[ix0+1, iy0,   iz0]
                    c010 = d_volume[ix0,   iy0+1, iz0]
                    c001 = d_volume[ix0,   iy0,   iz0+1]
                    c110 = d_volume[ix0+1, iy0+1, iz0]
                    c101 = d_volume[ix0+1, iy0,   iz0+1]
                    c011 = d_volume[ix0,   iy0+1, iz0+1]
                    c111 = d_volume[ix0+1, iy0+1, iz0+1]
                    val = (c000*(1-dx2)*(1-dy2)*(1-dz2) +
                           c100*dx2*(1-dy2)*(1-dz2) +
                           c010*(1-dx2)*dy2*(1-dz2) +
                           c001*(1-dx2)*(1-dy2)*dz2 +
                           c110*dx2*dy2*(1-dz2) +
                           c101*dx2*(1-dy2)*dz2 +
                           c011*(1-dx2)*dy2*dz2 +
                           c111*dx2*dy2*dz2)
                    total_val += val*step_size
                t += step_size

            d_sino[iview, iu, ivz] = total_val

    @staticmethod
    def backward(ctx, grad_sinogram):
        (volume, angles) = ctx.saved_tensors
        (Nx, Ny, Nz, det_u, det_v, du, dv, step_size, source_distance, isocenter_distance) = ctx.intermediate
        device = volume.device

        d_grad_sino = cuda.to_device(grad_sinogram.detach().cpu().numpy())
        d_volume_grad = cuda.device_array((Nx, Ny, Nz), dtype=np.float32)
        d_volume_grad[:] = 0.0

        d_angles = cuda.to_device(angles.detach().cpu().numpy())
        num_views = angles.shape[0]

        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(num_views / threadsperblock[0])
        blockspergrid_y = math.ceil(det_u / threadsperblock[1])
        blockspergrid_z = math.ceil(det_v / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5
        cz = (Nz - 1)*0.5

        ConeProjectorFunction._backward_kernel[blockspergrid, threadsperblock](
            d_grad_sino, num_views, det_u, det_v,
            Nx, Ny, Nz, d_volume_grad,
            du, dv, d_angles, step_size,
            source_distance, isocenter_distance,
            cx, cy, cz
        )

        grad_volume = torch.tensor(d_volume_grad.copy_to_host(), device=device)
        return grad_volume, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    @cuda.jit
    def _backward_kernel(d_grad_sino, num_views, det_u, det_v,
                         Nx, Ny, Nz, d_vol_grad,
                         du, dv, d_angles, step_size,
                         source_distance, isocenter_distance,
                         cx, cy, cz):
        iview, iu, ivz = cuda.grid(3)
        if iview < num_views and iu < det_u and ivz < det_v:
            grad_val = d_grad_sino[iview, iu, ivz]
            angle = d_angles[iview]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (iu - (det_u - 1)*0.5)*du
            v = (ivz - (det_v - 1)*0.5)*dv

            sx = -isocenter_distance*sin_a
            sy =  isocenter_distance*cos_a
            sz =  0.0

            dx = (source_distance - isocenter_distance)*sin_a + u*cos_a
            dy = -(source_distance - isocenter_distance)*cos_a + u*sin_a
            dz = v

            rx = dx - sx
            ry = dy - sy
            rz = dz - sz
            length = math.sqrt(rx*rx + ry*ry + rz*rz)
            rx /= length
            ry /= length
            rz /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t*rx
                y = sy + t*ry
                z = sz + t*rz
                ix = x + cx
                iy = y + cy
                iz = z + cz
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))
                iz0 = int(math.floor(iz))
                if (ix0 >= 0 and ix0 < Nx-1 and
                    iy0 >= 0 and iy0 < Ny-1 and
                    iz0 >= 0 and iz0 < Nz-1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    dz2 = iz - iz0
                    cval = grad_val*step_size
                    cuda.atomic.add(d_vol_grad, (ix0,   iy0,   iz0),   cval*(1-dx2)*(1-dy2)*(1-dz2))
                    cuda.atomic.add(d_vol_grad, (ix0+1, iy0,   iz0),   cval*dx2*(1-dy2)*(1-dz2))
                    cuda.atomic.add(d_vol_grad, (ix0,   iy0+1, iz0),   cval*(1-dx2)*dy2*(1-dz2))
                    cuda.atomic.add(d_vol_grad, (ix0,   iy0,   iz0+1), cval*(1-dx2)*(1-dy2)*dz2)
                    cuda.atomic.add(d_vol_grad, (ix0+1, iy0+1, iz0),   cval*dx2*dy2*(1-dz2))
                    cuda.atomic.add(d_vol_grad, (ix0+1, iy0,   iz0+1), cval*dx2*(1-dy2)*dz2)
                    cuda.atomic.add(d_vol_grad, (ix0,   iy0+1, iz0+1), cval*(1-dx2)*dy2*dz2)
                    cuda.atomic.add(d_vol_grad, (ix0+1, iy0+1, iz0+1), cval*dx2*dy2*dz2)
                t += step_size

############################################################################
# Cone Beam: Differentiable Backprojector
############################################################################

class ConeBackprojectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles, Nx, Ny, Nz, det_u, det_v, du, dv, step_size, source_distance, isocenter_distance):
        device = sinogram.device
        num_views = angles.shape[0]

        d_sino = cuda.to_device(sinogram.detach().cpu().numpy())
        d_image = cuda.device_array((Nx, Ny, Nz), dtype=np.float32)
        d_image[:] = 0.0
        d_angles = cuda.to_device(angles.detach().cpu().numpy())

        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(num_views / threadsperblock[0])
        blockspergrid_y = math.ceil(det_u / threadsperblock[1])
        blockspergrid_z = math.ceil(det_v / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5
        cz = (Nz - 1)*0.5

        ConeBackprojectorFunction._forward_kernel[blockspergrid, threadsperblock](
            d_sino, num_views, det_u, det_v,
            Nx, Ny, Nz, d_image,
            du, dv, d_angles,
            step_size,
            source_distance,
            isocenter_distance,
            cx, cy, cz
        )

        volume = torch.tensor(d_image.copy_to_host(), device=device)
        ctx.save_for_backward(sinogram, angles)
        ctx.intermediate = (Nx, Ny, Nz, det_u, det_v, du, dv, step_size, source_distance, isocenter_distance)
        return volume

    @staticmethod
    @cuda.jit
    def _forward_kernel(d_sino, num_views, det_u, det_v,
                        Nx, Ny, Nz, d_reco,
                        du, dv, d_angles,
                        step_size,
                        source_distance,
                        isocenter_distance,
                        cx, cy, cz):
        iview, iu, ivz = cuda.grid(3)
        if iview < num_views and iu < det_u and ivz < det_v:
            val = d_sino[iview, iu, ivz]
            angle = d_angles[iview]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (iu - (det_u - 1)*0.5)*du
            v = (ivz - (det_v - 1)*0.5)*dv

            sx = -isocenter_distance*sin_a
            sy =  isocenter_distance*cos_a
            sz =  0.0

            dx = (source_distance - isocenter_distance)*sin_a + u*cos_a
            dy = -(source_distance - isocenter_distance)*cos_a + u*sin_a
            dz = v

            rx = dx - sx
            ry = dy - sy
            rz = dz - sz
            length = math.sqrt(rx*rx + ry*ry + rz*rz)
            rx /= length
            ry /= length
            rz /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t*rx
                y = sy + t*ry
                z = sz + t*rz

                ix = x + cx
                iy = y + cy
                iz = z + cz
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))
                iz0 = int(math.floor(iz))
                if (ix0 >= 0 and ix0 < Nx-1 and
                    iy0 >= 0 and iy0 < Ny-1 and
                    iz0 >= 0 and iz0 < Nz-1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    dz2 = iz - iz0
                    cval = val*step_size
                    cuda.atomic.add(d_reco, (ix0,   iy0,   iz0),   cval*(1-dx2)*(1-dy2)*(1-dz2))
                    cuda.atomic.add(d_reco, (ix0+1, iy0,   iz0),   cval*dx2*(1-dy2)*(1-dz2))
                    cuda.atomic.add(d_reco, (ix0,   iy0+1, iz0),   cval*(1-dx2)*dy2*(1-dz2))
                    cuda.atomic.add(d_reco, (ix0,   iy0,   iz0+1), cval*(1-dx2)*(1-dy2)*dz2)
                    cuda.atomic.add(d_reco, (ix0+1, iy0+1, iz0),   cval*dx2*dy2*(1-dz2))
                    cuda.atomic.add(d_reco, (ix0+1, iy0,   iz0+1), cval*dx2*(1-dy2)*dz2)
                    cuda.atomic.add(d_reco, (ix0,   iy0+1, iz0+1), cval*(1-dx2)*dy2*dz2)
                    cuda.atomic.add(d_reco, (ix0+1, iy0+1, iz0+1), cval*dx2*dy2*dz2)
                t += step_size

    @staticmethod
    def backward(ctx, grad_output):
        (sinogram, angles) = ctx.saved_tensors
        (Nx, Ny, Nz, det_u, det_v, du, dv, step_size, source_distance, isocenter_distance) = ctx.intermediate
        device = sinogram.device

        d_grad_out = cuda.to_device(grad_output.detach().cpu().numpy())
        d_sino_grad = cuda.device_array((sinogram.shape[0], det_u, det_v), dtype=np.float32)
        d_sino_grad[:] = 0.0

        d_angles = cuda.to_device(angles.detach().cpu().numpy())
        num_views = angles.shape[0]

        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(num_views / threadsperblock[0])
        blockspergrid_y = math.ceil(det_u / threadsperblock[1])
        blockspergrid_z = math.ceil(det_v / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5
        cz = (Nz - 1)*0.5

        ConeBackprojectorFunction._backward_kernel[blockspergrid, threadsperblock](
            d_grad_out, num_views, det_u, det_v,
            Nx, Ny, Nz,
            d_sino_grad,
            du, dv, d_angles,
            step_size,
            source_distance,
            isocenter_distance,
            cx, cy, cz
        )

        grad_sino = torch.tensor(d_sino_grad.copy_to_host(), device=device)
        return grad_sino, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    @cuda.jit
    def _backward_kernel(d_grad_out, num_views, det_u, det_v,
                         Nx, Ny, Nz,
                         d_sino_grad,
                         du, dv, d_angles,
                         step_size,
                         source_distance,
                         isocenter_distance,
                         cx, cy, cz):
        iview, iu, ivz = cuda.grid(3)
        if iview < num_views and iu < det_u and ivz < det_v:
            angle = d_angles[iview]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (iu - (det_u - 1)*0.5)*du
            v = (ivz - (det_v - 1)*0.5)*dv

            sx = -isocenter_distance*sin_a
            sy =  isocenter_distance*cos_a
            sz =  0.0

            dx = (source_distance - isocenter_distance)*sin_a + u*cos_a
            dy = -(source_distance - isocenter_distance)*cos_a + u*sin_a
            dz = v

            rx = dx - sx
            ry = dy - sy
            rz = dz - sz
            length = math.sqrt(rx*rx + ry*ry + rz*rz)
            rx /= length
            ry /= length
            rz /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t*rx
                y = sy + t*ry
                z = sz + t*rz
                ix = x + cx
                iy = y + cy
                iz = z + cz
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))
                iz0 = int(math.floor(iz))
                if (ix0 >= 0 and ix0 < Nx-1 and
                    iy0 >= 0 and iy0 < Ny-1 and
                    iz0 >= 0 and iz0 < Nz-1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    dz2 = iz - iz0
                    cval = (d_grad_out[ix0,   iy0,   iz0] * (1-dx2)*(1-dy2)*(1-dz2) +
                            d_grad_out[ix0+1, iy0,   iz0] * dx2*(1-dy2)*(1-dz2) +
                            d_grad_out[ix0,   iy0+1, iz0] * (1-dx2)*dy2*(1-dz2) +
                            d_grad_out[ix0,   iy0,   iz0+1] * (1-dx2)*(1-dy2)*dz2 +
                            d_grad_out[ix0+1, iy0+1, iz0] * dx2*dy2*(1-dz2) +
                            d_grad_out[ix0+1, iy0,   iz0+1] * dx2*(1-dy2)*dz2 +
                            d_grad_out[ix0,   iy0+1, iz0+1] * (1-dx2)*dy2*dz2 +
                            d_grad_out[ix0+1, iy0+1, iz0+1] * dx2*dy2*dz2) * step_size
                    cuda.atomic.add(d_sino_grad, (iview, iu, ivz), cval)
                t += step_size

def shepp_logan_3d(shape):
    shepp_logan = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
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
        mask = (xp**2)/(a_axis*a_axis) + (yp**2)/(b_axis*b_axis) + (zp**2)/(c_axis*c_axis) <= 1.0
        shepp_logan[mask] += val
    shepp_logan = np.clip(shepp_logan, 0, 1)
    return shepp_logan

def ramp_filter_3d(sinogram):
    num_views, num_det_u, num_det_v = sinogram.shape
    freqs = np.fft.fftfreq(num_det_u)
    omega = 2.0 * np.pi * freqs
    ramp = np.abs(omega)
    ramp_2d = ramp.reshape(1, num_det_u, 1)
    sino_fft = np.fft.fft(sinogram, axis=1)
    filtered_fft = sino_fft * ramp_2d
    filtered = np.real(np.fft.ifft(filtered_fft, axis=1))
    return filtered

def example_cone_pipeline():
    Nx, Ny, Nz = 128, 128, 128
    phantom_np = shepp_logan_3d((Nx, Ny, Nz))

    num_views = 180
    angles_np = np.linspace(0, 2*math.pi, num_views, endpoint=False).astype(np.float32)

    det_u, det_v = 256, 256
    du, dv = 1.0, 1.0
    step_size = 1.0
    source_distance = 600.0
    isocenter_distance = 400.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phantom_torch = torch.tensor(phantom_np, device=device, requires_grad=True)
    angles_torch = torch.tensor(angles_np, device=device)

    sinogram = ConeProjectorFunction.apply(phantom_torch, angles_torch, Nx, Ny, Nz,
                                           det_u, det_v, du, dv, step_size,
                                           source_distance, isocenter_distance)

    sinogram_np = sinogram.detach().cpu().numpy()
    sinogram_filt = ramp_filter_3d(sinogram_np)
    sinogram_filt = torch.tensor(sinogram_filt, device=device).contiguous()

    reconstruction = ConeBackprojectorFunction.apply(sinogram_filt, angles_torch,
                                                     Nx, Ny, Nz, det_u, det_v, du, dv,
                                                     step_size, source_distance, isocenter_distance)

    loss = torch.mean((reconstruction - phantom_torch)**2)
    loss.backward()

    print("Cone Beam Example with user-defined geometry:")
    print("Loss:", loss.item())
    print("Volume center voxel gradient:", phantom_torch.grad[Nx//2, Ny//2, Nz//2].item())
    print("Reconstruction shape:", reconstruction.shape)

    reconstruction_cpu = reconstruction.detach().cpu().numpy()
    phantom_cpu = phantom_np
    mid_slice = Nz // 2

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(phantom_cpu[:,:,mid_slice], cmap='gray')
    plt.title("Phantom mid-slice")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(sinogram_np[num_views//2], cmap='gray')
    plt.title("Sinogram mid-view")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(reconstruction_cpu[:,:,mid_slice], cmap='gray')
    plt.title("Recon mid-slice")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    example_cone_pipeline()