import math
import numpy as np
import torch
from numba import cuda

############################################################################
# Parallel Beam: Differentiable Forward Projector
############################################################################

class ParallelProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, angles, num_detectors, detector_spacing=1.0, step_size=0.5):
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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5

        ParallelProjectorFunction._forward_kernel[blockspergrid, threadsperblock](
            d_image, Nx, Ny,
            d_sinogram, num_angles, num_detectors,
            detector_spacing, d_angles, step_size,
            cx, cy
        )

        sinogram = torch.tensor(d_sinogram.copy_to_host(), device=device)
        ctx.save_for_backward(image, angles)
        ctx.intermediate = (num_detectors, detector_spacing, step_size, Nx, Ny)
        return sinogram

    @staticmethod
    @cuda.jit(fastmath=True)
    def _forward_kernel(d_image, Nx, Ny,
                        d_sinogram, num_angles, num_detectors,
                        detector_spacing, d_angles, step_size,
                        cx, cy):
        iang, idet = cuda.grid(2)
        if iang < num_angles and idet < num_detectors:
            angle = d_angles[iang]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (idet - (num_detectors - 1)*0.5)*detector_spacing
            t_min = -math.sqrt(Nx*Nx + Ny*Ny)  # Optimized bound
            t_max = math.sqrt(Nx*Nx + Ny*Ny)
            t = t_min

            total_val = 0.0
            while t < t_max:
                x = u*(-sin_a) + t*cos_a
                y = u*cos_a + t*sin_a

                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))

                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1):
                    dx = ix - ix0
                    dy = iy - iy0
                    c00 = d_image[ix0,     iy0]
                    c10 = d_image[ix0 + 1, iy0]
                    c01 = d_image[ix0,     iy0 + 1]
                    c11 = d_image[ix0 + 1, iy0 + 1]
                    val = (c00*(1-dx)*(1-dy) +
                           c10*dx*(1-dy) +
                           c01*(1-dx)*dy +
                           c11*dx*dy)
                    total_val += val * step_size

                t += step_size

            d_sinogram[iang, idet] = total_val

    @staticmethod
    def backward(ctx, grad_sinogram):
        (image, angles) = ctx.saved_tensors
        (num_detectors, detector_spacing, step_size, Nx, Ny) = ctx.intermediate
        device = image.device

        d_grad_sino = cuda.to_device(grad_sinogram.detach().cpu().numpy())
        d_image_grad = cuda.device_array((Nx, Ny), dtype=np.float32)
        d_image_grad[:] = 0

        d_angles = cuda.to_device(angles.detach().cpu().numpy())

        num_angles = angles.shape[0]

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(num_angles / threadsperblock[0])
        blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5

        ParallelProjectorFunction._backward_kernel[blockspergrid, threadsperblock](
            d_grad_sino, num_angles, num_detectors,
            Nx, Ny, d_image_grad,
            detector_spacing, d_angles, step_size,
            cx, cy
        )

        grad_image = torch.tensor(d_image_grad.copy_to_host(), device=device)
        return grad_image, None, None, None, None

    @staticmethod
    @cuda.jit(fastmath=True)
    def _backward_kernel(d_grad_sino, num_angles, num_detectors,
                         Nx, Ny, d_image_grad,
                         detector_spacing, d_angles, step_size,
                         cx, cy):
        iang, idet = cuda.grid(2)
        if iang < num_angles and idet < num_detectors:
            grad_val = d_grad_sino[iang, idet]
            angle = d_angles[iang]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (idet - (num_detectors - 1)*0.5)*detector_spacing
            t_min = -math.sqrt(Nx*Nx + Ny*Ny)
            t_max = math.sqrt(Nx*Nx + Ny*Ny)
            t = t_min
            while t < t_max:
                x = u*(-sin_a) + t*cos_a
                y = u*cos_a + t*sin_a

                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))

                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1):
                    dx = ix - ix0
                    dy = iy - iy0
                    cval = grad_val*step_size
                    cuda.atomic.add(d_image_grad, (ix0, iy0),     cval*(1-dx)*(1-dy))
                    cuda.atomic.add(d_image_grad, (ix0+1, iy0),   cval*dx*(1-dy))
                    cuda.atomic.add(d_image_grad, (ix0, iy0+1),   cval*(1-dx)*dy)
                    cuda.atomic.add(d_image_grad, (ix0+1, iy0+1), cval*dx*dy)

                t += step_size

############################################################################
# Parallel Beam: Differentiable Backprojector
############################################################################

class ParallelBackprojectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles, detector_spacing=1.0, step_size=0.5,
                Nx=128, Ny=128):
        device = sinogram.device
        num_angles, num_detectors = sinogram.shape

        d_sino = cuda.to_device(sinogram.detach().cpu().numpy())
        d_image = cuda.device_array((Nx, Ny), dtype=np.float32)
        d_image[:] = 0

        d_angles = cuda.to_device(angles.detach().cpu().numpy())

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(num_angles / threadsperblock[0])
        blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5

        ParallelBackprojectorFunction._forward_kernel[blockspergrid, threadsperblock](
            d_sino, num_angles, num_detectors,
            Nx, Ny, d_image,
            detector_spacing, d_angles, step_size,
            cx, cy
        )

        reco = torch.tensor(d_image.copy_to_host(), device=device)
        ctx.save_for_backward(sinogram, angles)
        ctx.intermediate = (Nx, Ny, detector_spacing, step_size)
        return reco

    @staticmethod
    @cuda.jit(fastmath=True)
    def _forward_kernel(d_sino, num_angles, num_detectors,
                        Nx, Ny, d_image,
                        detector_spacing, d_angles, step_size,
                        cx, cy):
        iang, idet = cuda.grid(2)
        if iang < num_angles and idet < num_detectors:
            val = d_sino[iang, idet]
            angle = d_angles[iang]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (idet - (num_detectors - 1)*0.5)*detector_spacing
            t_min = -math.sqrt(Nx*Nx + Ny*Ny)
            t_max = math.sqrt(Nx*Nx + Ny*Ny)
            t = t_min
            while t < t_max:
                x = u*(-sin_a) + t*cos_a
                y = u*cos_a + t*sin_a

                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))

                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1):
                    dx = ix - ix0
                    dy = iy - iy0
                    cval = val*step_size
                    cuda.atomic.add(d_image, (ix0, iy0),     cval*(1-dx)*(1-dy))
                    cuda.atomic.add(d_image, (ix0+1, iy0),   cval*dx*(1-dy))
                    cuda.atomic.add(d_image, (ix0, iy0+1),   cval*(1-dx)*dy)
                    cuda.atomic.add(d_image, (ix0+1, iy0+1), cval*dx*dy)
                t += step_size

    @staticmethod
    def backward(ctx, grad_output):
        (sinogram, angles) = ctx.saved_tensors
        Nx, Ny, detector_spacing, step_size = ctx.intermediate
        device = sinogram.device

        d_grad_out = cuda.to_device(grad_output.detach().cpu().numpy())
        num_angles, num_detectors = sinogram.shape

        d_sino_grad = cuda.device_array((num_angles, num_detectors), dtype=np.float32)
        d_sino_grad[:] = 0

        d_angles = cuda.to_device(angles.detach().cpu().numpy())

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(num_angles / threadsperblock[0])
        blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cx = (Nx - 1)*0.5
        cy = (Ny - 1)*0.5

        ParallelBackprojectorFunction._backward_kernel[blockspergrid, threadsperblock](
            d_grad_out, num_angles, num_detectors,
            Nx, Ny, d_sino_grad,
            detector_spacing, d_angles, step_size,
            cx, cy
        )

        grad_sino = torch.tensor(d_sino_grad.copy_to_host(), device=device)
        return grad_sino, None, None, None, None, None

    @staticmethod
    @cuda.jit(fastmath=True)
    def _backward_kernel(d_grad_out, num_angles, num_detectors,
                         Nx, Ny, d_sino_grad,
                         detector_spacing, d_angles, step_size,
                         cx, cy):
        iang, idet = cuda.grid(2)
        if iang < num_angles and idet < num_detectors:
            angle = d_angles[iang]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            u = (idet - (num_detectors - 1)*0.5)*detector_spacing
            t_min = -math.sqrt(Nx*Nx + Ny*Ny)
            t_max = math.sqrt(Nx*Nx + Ny*Ny)
            t = t_min
            while t < t_max:
                x = u*(-sin_a) + t*cos_a
                y = u*cos_a + t*sin_a

                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))

                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1):
                    dx = ix - ix0
                    dy = iy - iy0
                    cval = d_grad_out[ix0, iy0] * step_size * (1-dx)*(1-dy) + \
                           d_grad_out[ix0+1, iy0] * step_size * dx*(1-dy) + \
                           d_grad_out[ix0, iy0+1] * step_size * (1-dx)*dy + \
                           d_grad_out[ix0+1, iy0+1] * step_size * dx*dy
                    cuda.atomic.add(d_sino_grad, (iang, idet), cval)
                t += step_size

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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5

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
    @cuda.jit(fastmath=True)
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

            u = (idet - (num_detectors - 1) * 0.5) * det_spacing
            sx = -isocenter_distance * sin_a
            sy = isocenter_distance * cos_a
            dx = (source_distance - isocenter_distance) * sin_a + u * cos_a
            dy = -(source_distance - isocenter_distance) * cos_a + u * sin_a

            rx = dx - sx
            ry = dy - sy
            length = math.sqrt(rx * rx + ry * ry)
            rx /= length
            ry /= length

            total_val = 0.0
            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t * rx
                y = sy + t * ry
                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))
                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0
                    c00 = d_image[ix0, iy0]
                    c10 = d_image[ix0 + 1, iy0]
                    c01 = d_image[ix0, iy0 + 1]
                    c11 = d_image[ix0 + 1, iy0 + 1]
                    val = (c00 * (1 - dx2) * (1 - dy2) +
                           c10 * dx2 * (1 - dy2) +
                           c01 * (1 - dx2) * dy2 +
                           c11 * dx2 * dy2)
                    total_val += val * step_size
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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5

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
    @cuda.jit(fastmath=True)
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
            u = (idet - (num_detectors - 1) * 0.5) * det_spacing
            sx = -isocenter_distance * sin_a
            sy = isocenter_distance * cos_a
            dx = (source_distance - isocenter_distance) * sin_a + u * cos_a
            dy = -(source_distance - isocenter_distance) * cos_a + u * sin_a
            rx = dx - sx
            ry = dy - sy
            length = math.sqrt(rx * rx + ry * ry)
            rx /= length
            ry /= length
            
            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t * rx
                y = sy + t * ry
                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))
                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1):
                    dx = ix - ix0
                    dy = iy - iy0
                    cval = grad_val * step_size
                    cuda.atomic.add(d_image_grad, (ix0, iy0), cval * (1 - dx) * (1 - dy))
                    cuda.atomic.add(d_image_grad, (ix0 + 1, iy0), cval * dx * (1 - dy))
                    cuda.atomic.add(d_image_grad, (ix0, iy0 + 1), cval * (1 - dx) * dy)
                    cuda.atomic.add(d_image_grad, (ix0 + 1, iy0 + 1), cval * dx * dy)
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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5

        FanBackprojectorFunction._forward_kernel[blockspergrid, threadsperblock](
            d_sino, num_angles, num_detectors,
            Nx, Ny, d_image,
            detector_spacing, d_angles,
            step_size,
            source_distance,
            isocenter_distance,
            cx, cy
        )

        image = torch.tensor(d_image.copy_to_host(), device=device)
        ctx.save_for_backward(sinogram, angles)
        ctx.intermediate = (Nx, Ny, detector_spacing, step_size, source_distance, isocenter_distance)
        return image

    @staticmethod
    @cuda.jit(fastmath=True)
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
            u = (idet - (num_detectors - 1) * 0.5) * det_spacing
            sx = -isocenter_distance * sin_a
            sy = isocenter_distance * cos_a
            dx = (source_distance - isocenter_distance) * sin_a + u * cos_a
            dy = -(source_distance - isocenter_distance) * cos_a + u * sin_a
            detector_pos_u = (idet - (num_detectors - 1) / 2.0) * det_spacing

            rx = dx - sx
            ry = dy - sy
            length = math.sqrt(rx * rx + ry * ry)
            rx /= length
            ry /= length
            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t * rx
                y = sy + t * ry
                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))
                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1):
                    dx = ix - ix0
                    dy = iy - iy0

                    cval = val * step_size
                    cuda.atomic.add(d_image, (ix0, iy0), cval * (1 - dx) * (1 - dy))
                    cuda.atomic.add(d_image, (ix0 + 1, iy0), cval * dx * (1 - dy))
                    cuda.atomic.add(d_image, (ix0, iy0 + 1), cval * (1 - dx) * dy)
                    cuda.atomic.add(d_image, (ix0 + 1, iy0 + 1), cval * dx * dy)
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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5

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
    @cuda.jit(fastmath=True)
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

            u = (idet - (num_detectors - 1) * 0.5) * det_spacing
            sx = -isocenter_distance * sin_a
            sy = isocenter_distance * cos_a
            dx = (source_distance - isocenter_distance) * sin_a + u * cos_a
            dy = -(source_distance - isocenter_distance) * cos_a + u * sin_a

            rx = dx - sx
            ry = dy - sy
            length = math.sqrt(rx * rx + ry * ry)
            rx /= length
            ry /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t * rx
                y = sy + t * ry
                ix = x + cx
                iy = y + cy
                ix0 = int(math.floor(ix))
                iy0 = int(math.floor(iy))

                if (ix0 >= 0 and ix0 < Nx - 1 and
                    iy0 >= 0 and iy0 < Ny - 1):
                    dx2 = ix - ix0
                    dy2 = iy - iy0

                    cval = (d_grad_out[ix0, iy0] * (1 - dx2) * (1 - dy2) +
                            d_grad_out[ix0 + 1, iy0] * dx2 * (1 - dy2) +
                            d_grad_out[ix0, iy0 + 1] * (1 - dx2) * dy2 +
                            d_grad_out[ix0 + 1, iy0 + 1] * dx2 * dy2) * step_size

                    cuda.atomic.add(d_sino_grad, (iang, idet), cval)
                t += step_size

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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5
        cz = (Nz - 1) * 0.5

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
    @cuda.jit(fastmath=True)
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

            u = (iu - (det_u - 1) * 0.5) * du
            v = (ivz - (det_v - 1) * 0.5) * dv

            sx = -isocenter_distance * sin_a
            sy = isocenter_distance * cos_a
            sz = 0.0

            dx = (source_distance - isocenter_distance) * sin_a + u * cos_a
            dy = -(source_distance - isocenter_distance) * cos_a + u * sin_a
            dz = v

            rx = dx - sx
            ry = dy - sy
            rz = dz - sz
            length = math.sqrt(rx * rx + ry * ry + rz * rz)
            rx /= length
            ry /= length
            rz /= length

            total_val = 0.0
            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t * rx
                y = sy + t * ry
                z = sz + t * rz

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
                    c000 = d_volume[ix0, iy0, iz0]
                    c100 = d_volume[ix0 + 1, iy0, iz0]
                    c010 = d_volume[ix0, iy0 + 1, iz0]
                    c001 = d_volume[ix0, iy0, iz0 + 1]
                    c110 = d_volume[ix0 + 1, iy0 + 1, iz0]
                    c101 = d_volume[ix0 + 1, iy0, iz0 + 1]
                    c011 = d_volume[ix0, iy0 + 1, iz0 + 1]
                    c111 = d_volume[ix0 + 1, iy0 + 1, iz0 + 1]
                    val = (c000 * (1 - dx2) * (1 - dy2) * (1 - dz2) +
                           c100 * dx2 * (1 - dy2) * (1 - dz2) +
                           c010 * (1 - dx2) * dy2 * (1 - dz2) +
                           c001 * (1 - dx2) * (1 - dy2) * dz2 +
                           c110 * dx2 * dy2 * (1 - dz2) +
                           c101 * dx2 * (1 - dy2) * dz2 +
                           c011 * (1 - dx2) * dy2 * dz2 +
                           c111 * dx2 * dy2 * dz2)
                    total_val += val * step_size
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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5
        cz = (Nz - 1) * 0.5

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
    @cuda.jit(fastmath=True)
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

            u = (iu - (det_u - 1) * 0.5) * du
            v = (ivz - (det_v - 1) * 0.5) * dv

            sx = -isocenter_distance * sin_a
            sy = isocenter_distance * cos_a
            sz = 0.0

            dx = (source_distance - isocenter_distance) * sin_a + u * cos_a
            dy = -(source_distance - isocenter_distance) * cos_a + u * sin_a
            dz = v

            rx = dx - sx
            ry = dy - sy
            rz = dz - sz
            length = math.sqrt(rx * rx + ry * ry + rz * rz)
            rx /= length
            ry /= length
            rz /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t * rx
                y = sy + t * ry
                z = sz + t * rz
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
                    cval = grad_val * step_size
                    cuda.atomic.add(d_vol_grad, (ix0, iy0, iz0), cval * (1 - dx2) * (1 - dy2) * (1 - dz2))
                    cuda.atomic.add(d_vol_grad, (ix0 + 1, iy0, iz0), cval * dx2 * (1 - dy2) * (1 - dz2))
                    cuda.atomic.add(d_vol_grad, (ix0, iy0 + 1, iz0), cval * (1 - dx2) * dy2 * (1 - dz2))
                    cuda.atomic.add(d_vol_grad, (ix0, iy0, iz0 + 1), cval * (1 - dx2) * (1 - dy2) * dz2)
                    cuda.atomic.add(d_vol_grad, (ix0 + 1, iy0 + 1, iz0), cval * dx2 * dy2 * (1 - dz2))
                    cuda.atomic.add(d_vol_grad, (ix0 + 1, iy0, iz0 + 1), cval * dx2 * (1 - dy2) * dz2)
                    cuda.atomic.add(d_vol_grad, (ix0, iy0 + 1, iz0 + 1), cval * (1 - dx2) * dy2 * dz2)
                    cuda.atomic.add(d_vol_grad, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx2 * dy2 * dz2)
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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5
        cz = (Nz - 1) * 0.5

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
    @cuda.jit(fastmath=True)
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

            u = (iu - (det_u - 1) * 0.5) * du
            v = (ivz - (det_v - 1) * 0.5) * dv

            sx = -isocenter_distance * sin_a
            sy = isocenter_distance * cos_a
            sz = 0.0

            dx = (source_distance - isocenter_distance) * sin_a + u * cos_a
            dy = -(source_distance - isocenter_distance) * cos_a + u * sin_a
            dz = v

            rx = dx - sx
            ry = dy - sy
            rz = dz - sz
            length = math.sqrt(rx * rx + ry * ry + rz * rz)
            rx /= length
            ry /= length
            rz /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t * rx
                y = sy + t * ry
                z = sz + t * rz

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

                    cval = val * step_size

                    cuda.atomic.add(d_reco, (ix0, iy0, iz0), cval * (1 - dx2) * (1 - dy2) * (1 - dz2))
                    cuda.atomic.add(d_reco, (ix0 + 1, iy0, iz0), cval * dx2 * (1 - dy2) * (1 - dz2))
                    cuda.atomic.add(d_reco, (ix0, iy0 + 1, iz0), cval * (1 - dx2) * dy2 * (1 - dz2))
                    cuda.atomic.add(d_reco, (ix0, iy0, iz0 + 1), cval * (1 - dx2) * (1 - dy2) * dz2)
                    cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0), cval * dx2 * dy2 * (1 - dz2))
                    cuda.atomic.add(d_reco, (ix0 + 1, iy0, iz0 + 1), cval * dx2 * (1 - dy2) * dz2)
                    cuda.atomic.add(d_reco, (ix0, iy0 + 1, iz0 + 1), cval * (1 - dx2) * dy2 * dz2)
                    cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx2 * dy2 * dz2)
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

        cx = (Nx - 1) * 0.5
        cy = (Ny - 1) * 0.5
        cz = (Nz - 1) * 0.5

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
    @cuda.jit(fastmath=True)
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

            u = (iu - (det_u - 1) * 0.5) * du
            v = (ivz - (det_v - 1) * 0.5) * dv

            sx = -isocenter_distance * sin_a
            sy = isocenter_distance * cos_a
            sz = 0.0

            dx = (source_distance - isocenter_distance) * sin_a + u * cos_a
            dy = -(source_distance - isocenter_distance) * cos_a + u * sin_a
            dz = v

            rx = dx - sx
            ry = dy - sy
            rz = dz - sz
            length = math.sqrt(rx * rx + ry * ry + rz * rz)
            rx /= length
            ry /= length
            rz /= length

            t_min = 0.0
            t_max = length
            t = t_min
            while t < t_max:
                x = sx + t * rx
                y = sy + t * ry
                z = sz + t * rz
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

                    cval = (d_grad_out[ix0, iy0, iz0] * (1 - dx2) * (1 - dy2) * (1 - dz2) +
                            d_grad_out[ix0 + 1, iy0, iz0] * dx2 * (1 - dy2) * (1 - dz2) +
                            d_grad_out[ix0, iy0 + 1, iz0] * (1 - dx2) * dy2 * (1 - dz2) +
                            d_grad_out[ix0, iy0, iz0 + 1] * (1 - dx2) * (1 - dy2) * dz2 +
                            d_grad_out[ix0 + 1, iy0 + 1, iz0] * dx2 * dy2 * (1 - dz2) +
                            d_grad_out[ix0 + 1, iy0, iz0 + 1] * dx2 * (1 - dy2) * dz2 +
                            d_grad_out[ix0, iy0 + 1, iz0 + 1] * (1 - dx2) * dy2 * dz2 +
                            d_grad_out[ix0 + 1, iy0 + 1, iz0 + 1] * dx2 * dy2 * dz2) * step_size
                    cuda.atomic.add(d_sino_grad, (iview, iu, ivz), cval)
                t += step_size