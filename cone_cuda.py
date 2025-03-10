import math
import numpy as np
from numba import cuda

@cuda.jit
def forward_cone_3d_kernel(
    d_volume, Nx, Ny, Nz, d_sinogram, num_views, num_det_u, num_det_v,
    du, dv, d_angles, source_distance, isocenter_distance, step_size,
    cx, cy, cz
):
    iview, iu, iv = cuda.grid(3)
    if iview < num_views and iu < num_det_u and iv < num_det_v:
        angle = d_angles[iview]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        u = (iu - (num_det_u - 1) / 2.0) * du
        v = (iv - (num_det_v - 1) / 2.0) * dv

        ray_source_x = -isocenter_distance * sin_a
        ray_source_y =  isocenter_distance * cos_a
        ray_source_z = 0.0

        ray_det_x = (source_distance - isocenter_distance) * sin_a + u * cos_a
        ray_det_y = -(source_distance - isocenter_distance) * cos_a + u * sin_a
        ray_det_z = v

        ray_dir_x = ray_det_x - ray_source_x
        ray_dir_y = ray_det_y - ray_source_y
        ray_dir_z = ray_det_z - ray_source_z
        ray_length = math.sqrt(ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2)

        ray_dir_x /= ray_length
        ray_dir_y /= ray_length
        ray_dir_z /= ray_length

        total_val = 0.0
        t_min = 0.0
        t_max = ray_length
        t = t_min
        while t < t_max:
            x = ray_source_x + t * ray_dir_x
            y = ray_source_y + t * ray_dir_y
            z = ray_source_z + t * ray_dir_z
            ix = x + cx
            iy = y + cy
            iz = z + cz
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            iz0 = int(math.floor(iz))

            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1 and 0 <= iz0 < Nz - 1:
                dx = ix - ix0
                dy = iy - iy0
                dz = iz - iz0

                c000 = d_volume[ix0,     iy0,     iz0]
                c100 = d_volume[ix0 + 1, iy0,     iz0]
                c010 = d_volume[ix0,     iy0 + 1, iz0]
                c001 = d_volume[ix0,     iy0,     iz0 + 1]
                c110 = d_volume[ix0 + 1, iy0 + 1, iz0]
                c101 = d_volume[ix0 + 1, iy0,     iz0 + 1]
                c011 = d_volume[ix0,     iy0 + 1, iz0 + 1]
                c111 = d_volume[ix0 + 1, iy0 + 1, iz0 + 1]

                val = (
                    c000 * (1 - dx) * (1 - dy) * (1 - dz)
                    + c100 * dx        * (1 - dy) * (1 - dz)
                    + c010 * (1 - dx)  * dy       * (1 - dz)
                    + c001 * (1 - dx)  * (1 - dy) * dz
                    + c110 * dx        * dy       * (1 - dz)
                    + c101 * dx        * (1 - dy) * dz
                    + c011 * (1 - dx)  * dy       * dz
                    + c111 * dx        * dy       * dz
                )
                total_val += val * step_size

            t += step_size

        d_sinogram[iview, iu, iv] = total_val

@cuda.jit
def back_cone_3d_kernel(
    d_sinogram, num_views, num_det_u, num_det_v, Nx, Ny, Nz,
    d_reco, du, dv, d_angles, source_distance, isocenter_distance,
    step_size, cx, cy, cz
):
    iview, iu, iv = cuda.grid(3)
    if iview < num_views and iu < num_det_u and iv < num_det_v:
        val = d_sinogram[iview, iu, iv]
        angle = d_angles[iview]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        u = (iu - (num_det_u - 1) / 2.0) * du
        v = (iv - (num_det_v - 1) / 2.0) * dv

        ray_source_x = -isocenter_distance * sin_a
        ray_source_y =  isocenter_distance * cos_a
        ray_source_z = 0.0

        ray_det_x = (source_distance - isocenter_distance) * sin_a + u * cos_a
        ray_det_y = -(source_distance - isocenter_distance) * cos_a + u * sin_a
        ray_det_z = v

        ray_dir_x = ray_det_x - ray_source_x
        ray_dir_y = ray_det_y - ray_source_y
        ray_dir_z = ray_det_z - ray_source_z
        ray_length = math.sqrt(ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2)

        ray_dir_x /= ray_length
        ray_dir_y /= ray_length
        ray_dir_z /= ray_length

        t_min = 0.0
        t_max = ray_length
        t = t_min
        while t < t_max:
            x = ray_source_x + t * ray_dir_x
            y = ray_source_y + t * ray_dir_y
            z = ray_source_z + t * ray_dir_z

            ix = x + cx
            iy = y + cy
            iz = z + cz
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            iz0 = int(math.floor(iz))

            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1 and 0 <= iz0 < Nz - 1:
                dx = ix - ix0
                dy = iy - iy0
                dz = iz - iz0

                cval = val * step_size

                cuda.atomic.add(d_reco, (ix0,     iy0,     iz0),     cval * (1 - dx) * (1 - dy) * (1 - dz))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0),     cval * dx       * (1 - dy) * (1 - dz))
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0),     cval * (1 - dx) * dy       * (1 - dz))
                cuda.atomic.add(d_reco, (ix0,     iy0,     iz0 + 1), cval * (1 - dx) * (1 - dy) * dz)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0),     cval * dx       * dy       * (1 - dz))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0 + 1), cval * dx       * (1 - dy) * dz)
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0 + 1), cval * (1 - dx) * dy       * dz)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx       * dy       * dz)

            t += step_size

def forward_cone_3d(
    volume, num_views, num_det_u, num_det_v, du, dv, angles,
    source_distance, isocenter_distance, step_size=0.5
):
    Nx, Ny, Nz = volume.shape
    d_volume = cuda.to_device(volume)
    d_angles = cuda.to_device(angles)
    d_sinogram = cuda.device_array((num_views, num_det_u, num_det_v), dtype=np.float64)

    threadsperblock = (8, 8, 8)
    blockspergrid_x = math.ceil(num_views / threadsperblock[0])
    blockspergrid_y = math.ceil(num_det_u / threadsperblock[1])
    blockspergrid_z = math.ceil(num_det_v / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    cx = (Nx - 1) / 2.0
    cy = (Ny - 1) / 2.0
    cz = (Nz - 1) / 2.0

    forward_cone_3d_kernel[blockspergrid, threadsperblock](
        d_volume, Nx, Ny, Nz, d_sinogram, num_views, num_det_u,
        num_det_v, du, dv, d_angles, source_distance,
        isocenter_distance, step_size, cx, cy, cz
    )
    return d_sinogram.copy_to_host()

def back_cone_3d(
    sinogram, Nx, Ny, Nz, du, dv, angles,
    source_distance, isocenter_distance, step_size=0.5
):
    num_views, num_det_u, num_det_v = sinogram.shape
    d_sinogram = cuda.to_device(sinogram)
    d_angles = cuda.to_device(angles)
    d_reco = cuda.device_array((Nx, Ny, Nz), dtype=np.float64)

    threadsperblock = (8, 8, 8)
    blockspergrid_x = math.ceil(num_views / threadsperblock[0])
    blockspergrid_y = math.ceil(num_det_u / threadsperblock[1])
    blockspergrid_z = math.ceil(num_det_v / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    cx = (Nx - 1) / 2.0
    cy = (Ny - 1) / 2.0
    cz = (Nz - 1) / 2.0

    back_cone_3d_kernel[blockspergrid, threadsperblock](
        d_sinogram, num_views, num_det_u, num_det_v, Nx, Ny, Nz,
        d_reco, du, dv, d_angles, source_distance,
        isocenter_distance, step_size, cx, cy, cz
    )
    return d_reco.copy_to_host()