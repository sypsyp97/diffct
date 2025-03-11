import math
import numpy as np
import torch
from numba import cuda
import matplotlib.pyplot as plt

############################################################################
# Fan Beam: Differentiable Forward Projector
############################################################################

class FanProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, angles, num_detectors, detector_spacing, step_size, source_distance, isocenter_distance):
        device = image.device
        Nx, Ny = image.shape
        num_angles = angles.shape[0]

        d_image = cuda.to_device(image.detach().cpu().numpy())
        d_angles = cuda.to_device(angles.detach().cpu().numpy())
        d_sinogram = cuda.device_array((num_angles, num_detectors), dtype=np.float32)

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(num_angles / threadsperblock[0])
        blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5

        FanProjectorFunction._forward_kernel[blockspergrid, threadsperblock](
            d_image, Nx, Ny,
            d_sinogram, num_angles, num_detectors,
            detector_spacing, d_angles,
            step_size,
            source_distance, 
            isocenter_distance,
            cx, cy
        )

        sinogram = torch.tensor(d_sinogram.copy_to_host(), device=device)
        ctx.save_for_backward(image, angles)
        ctx.intermediate = (num_detectors, detector_spacing, step_size, Nx, Ny, source_distance, isocenter_distance)
        return sinogram

    @staticmethod
    @cuda.jit
    def _forward_kernel(d_image, Nx, Ny,
                        d_sinogram, num_angles, num_detectors,
                        det_spacing, d_angles,
                        step_size,
                        source_distance,
                        isocenter_distance,
                        cx, cy):
        iang, idet = cuda.grid(2)
        if iang < num_angles and idet < num_detectors:
            angle = d_angles[iang]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (idet - (num_detectors - 1)*0.5)*det_spacing
            sx = -isocenter_distance * sin_a
            sy =  isocenter_distance  * cos_a
            dx = (source_distance - isocenter_distance)*sin_a + u*cos_a
            dy = -(source_distance - isocenter_distance)*cos_a + u*sin_a

            rx = dx - sx
            ry = dy - sy
            length = math.sqrt(rx*rx + ry*ry)
            rx /= length
            ry /= length

            total_val = 0.0
            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t*rx
                y = sy + t*ry
                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))
                if (ix0 >= 0 and ix0 < Nx-1 and
                    iy0 >= 0 and iy0 < Ny-1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    c00 = d_image[ix0, iy0]
                    c10 = d_image[ix0+1, iy0]
                    c01 = d_image[ix0,   iy0+1]
                    c11 = d_image[ix0+1, iy0+1]
                    val = (c00*(1-dx2)*(1-dy2) +
                           c10*dx2*(1-dy2) +
                           c01*(1-dx2)*dy2 +
                           c11*dx2*dy2)
                    total_val += val*step_size
                t += step_size

            d_sinogram[iang, idet] = total_val

    @staticmethod
    def backward(ctx, grad_sinogram):
        (image, angles) = ctx.saved_tensors
        (num_detectors, det_spacing, step_size, Nx, Ny, source_distance, isocenter_distance) = ctx.intermediate
        device = image.device

        d_grad_sino = cuda.to_device(grad_sinogram.detach().cpu().numpy())
        d_image_grad = cuda.device_array((Nx, Ny), dtype=np.float32)
        d_image_grad[:] = 0.0

        d_angles = cuda.to_device(angles.detach().cpu().numpy())
        num_angles = angles.shape[0]

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(num_angles / threadsperblock[0])
        blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5

        FanProjectorFunction._backward_kernel[blockspergrid, threadsperblock](
            d_grad_sino, num_angles, num_detectors,
            Nx, Ny, d_image_grad,
            det_spacing, d_angles,
            step_size,
            source_distance, 
            isocenter_distance,
            cx, cy
        )

        grad_image = torch.tensor(d_image_grad.copy_to_host(), device=device)
        return grad_image, None, None, None, None, None, None

    @staticmethod
    @cuda.jit
    def _backward_kernel(d_grad_sino, num_angles, num_detectors,
                         Nx, Ny, d_image_grad,
                         det_spacing, d_angles,
                         step_size,
                         source_distance, 
                         isocenter_distance,
                         cx, cy):
        iang, idet = cuda.grid(2)
        if iang < num_angles and idet < num_detectors:
            grad_val = d_grad_sino[iang, idet]
            angle = d_angles[iang]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (idet - (num_detectors - 1)*0.5)*det_spacing
            sx = -isocenter_distance * sin_a
            sy =  isocenter_distance  * cos_a
            dx = (source_distance - isocenter_distance)*sin_a + u*cos_a
            dy = -(source_distance - isocenter_distance)*cos_a + u*sin_a

            rx = dx - sx
            ry = dy - sy
            length = math.sqrt(rx*rx + ry*ry)
            rx /= length
            ry /= length

            t_min = 0.0
            t_max = length
            t = t_min

            while t < t_max:
                x = sx + t*rx
                y = sy + t*ry
                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))

                if (ix0 >= 0 and ix0 < Nx-1 and
                    iy0 >= 0 and iy0 < Ny-1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    cval = grad_val*step_size
                    cuda.atomic.add(d_image_grad, (ix0,   iy0),   cval*(1-dx2)*(1-dy2))
                    cuda.atomic.add(d_image_grad, (ix0+1, iy0),   cval*dx2*(1-dy2))
                    cuda.atomic.add(d_image_grad, (ix0,   iy0+1), cval*(1-dx2)*dy2)
                    cuda.atomic.add(d_image_grad, (ix0+1, iy0+1), cval*dx2*dy2)

                t += step_size

############################################################################
# Fan Beam: Differentiable Backprojector
############################################################################

class FanBackprojectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles, detector_spacing, step_size, Nx, Ny, source_distance, isocenter_distance):
        device = sinogram.device
        num_angles, num_detectors = sinogram.shape

        d_sino = cuda.to_device(sinogram.detach().cpu().numpy())
        d_image = cuda.device_array((Nx, Ny), dtype=np.float32)
        d_image[:] = 0.0

        d_angles = cuda.to_device(angles.detach().cpu().numpy())

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(num_angles / threadsperblock[0])
        blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5

        FanBackprojectorFunction._forward_kernel[blockspergrid, threadsperblock](
            d_sino, num_angles, num_detectors,
            Nx, Ny, d_image,
            detector_spacing, d_angles,
            step_size,
            source_distance, 
            isocenter_distance,
            cx, cy
        )

        reco = torch.tensor(d_image.copy_to_host(), device=device)
        ctx.save_for_backward(sinogram, angles)
        ctx.intermediate = (Nx, Ny, detector_spacing, step_size, source_distance, isocenter_distance)
        return reco

    @staticmethod
    @cuda.jit
    def _forward_kernel(d_sino, num_angles, num_detectors,
                        Nx, Ny, d_image,
                        det_spacing, d_angles,
                        step_size,
                        source_distance,
                        isocenter_distance,
                        cx, cy):
        iang, idet = cuda.grid(2)
        if iang < num_angles and idet < num_detectors:
            val = d_sino[iang, idet]
            angle = d_angles[iang]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (idet - (num_detectors - 1)*0.5)*det_spacing
            sx = -isocenter_distance * sin_a
            sy =  isocenter_distance  * cos_a
            dx = (source_distance - isocenter_distance)*sin_a + u*cos_a
            dy = -(source_distance - isocenter_distance)*cos_a + u*sin_a

            rx = dx - sx
            ry = dy - sy
            length = math.sqrt(rx*rx + ry*ry)
            rx /= length
            ry /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t*rx
                y = sy + t*ry
                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))

                if (ix0 >= 0 and ix0 < Nx-1 and
                    iy0 >= 0 and iy0 < Ny-1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    cval = val*step_size
                    cuda.atomic.add(d_image, (ix0,   iy0),   cval*(1-dx2)*(1-dy2))
                    cuda.atomic.add(d_image, (ix0+1, iy0),   cval*dx2*(1-dy2))
                    cuda.atomic.add(d_image, (ix0,   iy0+1), cval*(1-dx2)*dy2)
                    cuda.atomic.add(d_image, (ix0+1, iy0+1), cval*dx2*dy2)

                t += step_size

    @staticmethod
    def backward(ctx, grad_output):
        (sinogram, angles) = ctx.saved_tensors
        (Nx, Ny, det_spacing, step_size, source_distance, isocenter_distance) = ctx.intermediate
        device = sinogram.device

        d_grad_out = cuda.to_device(grad_output.detach().cpu().numpy())
        num_angles, num_detectors = sinogram.shape

        d_sino_grad = cuda.device_array((num_angles, num_detectors), dtype=np.float32)
        d_sino_grad[:] = 0.0

        d_angles = cuda.to_device(angles.detach().cpu().numpy())

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(num_angles / threadsperblock[0])
        blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5

        FanBackprojectorFunction._backward_kernel[blockspergrid, threadsperblock](
            d_grad_out, num_angles, num_detectors,
            Nx, Ny, d_sino_grad,
            det_spacing, d_angles, step_size,
            source_distance, 
            isocenter_distance,
            cx, cy
        )

        grad_sino = torch.tensor(d_sino_grad.copy_to_host(), device=device)
        return grad_sino, None, None, None, None, None, None, None

    @staticmethod
    @cuda.jit
    def _backward_kernel(d_grad_out, num_angles, num_detectors,
                         Nx, Ny, d_sino_grad,
                         det_spacing, d_angles, step_size,
                         source_distance, isocenter_distance,
                         cx, cy):
        iang, idet = cuda.grid(2)
        if iang < num_angles and idet < num_detectors:
            angle = d_angles[iang]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (idet - (num_detectors - 1)*0.5)*det_spacing
            sx = -isocenter_distance*sin_a
            sy =  isocenter_distance*cos_a
            dx = (source_distance - isocenter_distance)*sin_a + u*cos_a
            dy = -(source_distance - isocenter_distance)*cos_a + u*sin_a

            rx = dx - sx
            ry = dy - sy
            length = math.sqrt(rx*rx + ry*ry)
            rx /= length
            ry /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t*rx
                y = sy + t*ry
                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))

                if (ix0 >= 0 and ix0 < Nx-1 and
                    iy0 >= 0 and iy0 < Ny-1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    cval = (d_grad_out[ix0, iy0] * (1-dx2)*(1-dy2) +
                            d_grad_out[ix0+1, iy0] * dx2*(1-dy2) +
                            d_grad_out[ix0, iy0+1] * (1-dx2)*dy2 +
                            d_grad_out[ix0+1, iy0+1] * dx2*dy2) * step_size
                    cuda.atomic.add(d_sino_grad, (iang, idet), cval)
                t += step_size

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

def ramp_filter(sinogram):
    num_views, num_det = sinogram.shape
    freqs = np.fft.fftfreq(num_det)
    omega = 2.0 * np.pi * freqs
    ramp = np.abs(omega)
    ramp_2d = ramp.reshape(1, num_det)
    sino_fft = np.fft.fft(sinogram, axis=1)
    filtered_fft = sino_fft * ramp_2d
    filtered = np.real(np.fft.ifft(filtered_fft, axis=1))
    return filtered

def example_fan_pipeline():
    Nx, Ny = 256, 256
    phantom = shepp_logan_2d(Nx, Ny)
    num_angles = 360
    angles_np = np.linspace(0, 2*math.pi, num_angles, endpoint=False).astype(np.float32)

    num_detectors = 512
    detector_spacing = 1.0
    step_size = 0.5
    source_distance = 800.0
    isocenter_distance = 500.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_torch = torch.tensor(phantom, device=device, requires_grad=True)
    angles_torch = torch.tensor(angles_np, device=device)

    sinogram = FanProjectorFunction.apply(image_torch, angles_torch, num_detectors,
                                          detector_spacing, step_size, source_distance, isocenter_distance)

    sinogram_filt = ramp_filter(sinogram.detach().cpu().numpy())
    sinogram_filt = torch.tensor(sinogram_filt, device=device).contiguous()

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

if __name__ == "__main__":
    example_fan_pipeline()