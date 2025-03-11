import math
import numpy as np
from numba import cuda

@cuda.jit
def forward_parallel_2d_kernel(
    d_image, Nx, Ny, d_sinogram, num_views, num_detectors,
    detector_spacing, d_angles, step_size, cx, cy
):
    iview, idet = cuda.grid(2)
    if iview < num_views and idet < num_detectors:
        angle = d_angles[iview]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        u = (idet - (num_detectors - 1) / 2.0) * detector_spacing

        t_min = -math.sqrt(Nx*Nx + Ny*Ny)
        t_max = math.sqrt(Nx*Nx + Ny*Ny)
        total_val = 0.0
        t = t_min
        while t < t_max:
            x = u * (-sin_a) + t * cos_a
            y = u * cos_a + t * sin_a
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))

            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx = ix - ix0
                dy = iy - iy0
                val = (
                    d_image[ix0,     iy0]     * (1 - dx) * (1 - dy)
                    + d_image[ix0 + 1, iy0]   * dx       * (1 - dy)
                    + d_image[ix0,     iy0+1] * (1 - dx) * dy
                    + d_image[ix0 + 1, iy0+1] * dx       * dy
                )
                total_val += val * step_size
            t += step_size

        d_sinogram[iview, idet] = total_val

@cuda.jit
def back_parallel_2d_kernel(
    d_sinogram, Nx, Ny, d_reco, num_views, num_detectors,
    detector_spacing, d_angles, step_size, cx, cy
):
    iview, idet = cuda.grid(2)
    if iview < num_views and idet < num_detectors:
        val = d_sinogram[iview, idet]
        angle = d_angles[iview]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        u = (idet - (num_detectors - 1) / 2.0) * detector_spacing

        t_min = -math.sqrt(Nx*Nx + Ny*Ny)
        t_max = math.sqrt(Nx*Nx + Ny*Ny)
        t = t_min

        while t < t_max:
            x = u * (-sin_a) + t * cos_a
            y = u * cos_a + t * sin_a
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))

            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx = ix - ix0
                dy = iy - iy0
                cval = val * step_size
                cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1), cval * dx       * dy)
            t += step_size

def forward_parallel_2d(
    image, num_views, num_detectors, detector_spacing,
    angles, step_size=0.5
):
    Nx, Ny = image.shape
    d_image = cuda.to_device(image)
    d_angles = cuda.to_device(angles)
    d_sinogram = cuda.device_array((num_views, num_detectors), dtype=np.float64)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(num_views / threadsperblock[0])
    blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cx = (Nx - 1) / 2.0
    cy = (Ny - 1) / 2.0

    forward_parallel_2d_kernel[blockspergrid, threadsperblock](
        d_image, Nx, Ny, d_sinogram, num_views, num_detectors,
        detector_spacing, d_angles, step_size, cx, cy
    )
    return d_sinogram.copy_to_host()

def back_parallel_2d(
    sinogram, Nx, Ny, detector_spacing,
    angles, step_size=0.5
):
    num_views, num_detectors = sinogram.shape
    d_sinogram = cuda.to_device(sinogram)
    d_angles = cuda.to_device(angles)
    d_reco = cuda.device_array((Nx, Ny), dtype=np.float64)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(num_views / threadsperblock[0])
    blockspergrid_y = math.ceil(num_detectors / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cx = (Nx - 1) / 2.0
    cy = (Ny - 1) / 2.0

    back_parallel_2d_kernel[blockspergrid, threadsperblock](
        d_sinogram, Nx, Ny, d_reco, num_views, num_detectors,
        detector_spacing, d_angles, step_size, cx, cy
    )
    return d_reco.copy_to_host()