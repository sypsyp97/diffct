"""Metal kernels for 3D cone beam projections.

This module contains Metal kernel source strings implementing the Siddon
ray-tracing method for 3D cone beam forward projection and backprojection,
optimized for Apple Silicon GPUs.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants for Metal kernels
# ---------------------------------------------------------------------------
_EPSILON_STR = "1e-6f"
_INF_STR = "1e30f"

# ============================================================================
# 3D Cone Beam Forward Projection Metal Kernel
# ============================================================================

_CONE_3D_FORWARD_SOURCE = """
    // Thread position: (iview, iu, iv)
    uint iview = thread_position_in_grid.x;
    uint iu    = thread_position_in_grid.y;
    uint iv    = thread_position_in_grid.z;

    // Read scalar parameters
    int n_views = params[0];
    int n_u     = params[1];
    int n_v     = params[2];
    int Nx      = params[3];
    int Ny      = params[4];
    int Nz      = params[5];

    float du            = fparams[0];
    float dv            = fparams[1];
    float cx            = fparams[2];
    float cy            = fparams[3];
    float cz            = fparams[4];
    float voxel_spacing = fparams[5];

    if ((int)iview >= n_views || (int)iu >= n_u || (int)iv >= n_v) return;

    float eps = """ + _EPSILON_STR + """;

    // === 3D CONE BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iview * 3 + 0] / voxel_spacing;
    float src_y = src_pos[iview * 3 + 1] / voxel_spacing;
    float src_z = src_pos[iview * 3 + 2] / voxel_spacing;

    float det_cx = det_center[iview * 3 + 0] / voxel_spacing;
    float det_cy = det_center[iview * 3 + 1] / voxel_spacing;
    float det_cz = det_center[iview * 3 + 2] / voxel_spacing;

    float u_vec_x = det_u_vec_arr[iview * 3 + 0];
    float u_vec_y = det_u_vec_arr[iview * 3 + 1];
    float u_vec_z = det_u_vec_arr[iview * 3 + 2];

    float v_vec_x = det_v_vec_arr[iview * 3 + 0];
    float v_vec_y = det_v_vec_arr[iview * 3 + 1];
    float v_vec_z = det_v_vec_arr[iview * 3 + 2];

    float u_offset = ((float)iu - (float)n_u * 0.5f) * du / voxel_spacing;
    float v_offset = ((float)iv - (float)n_v * 0.5f) * dv / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x;
    float det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y;
    float det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z;

    // === 3D RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float dir_z = det_z - src_z;
    float length = metal::sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    if (length < eps) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;
    dir_z *= inv_len;

    // === 3D RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;

    if (metal::abs(dir_x) > eps) {
        float tx1 = (-cx - src_x) / dir_x;
        float tx2 = ( cx - src_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (src_x < -cx || src_x > cx) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (metal::abs(dir_y) > eps) {
        float ty1 = (-cy - src_y) / dir_y;
        float ty2 = ( cy - src_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (src_y < -cy || src_y > cy) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (metal::abs(dir_z) > eps) {
        float tz1 = (-cz - src_z) / dir_z;
        float tz2 = ( cz - src_z) / dir_z;
        t_min = metal::max(t_min, metal::min(tz1, tz2));
        t_max = metal::min(t_max, metal::max(tz1, tz2));
    } else if (src_z < -cz || src_z > cz) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (t_min >= t_max) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    // === 3D SIDDON TRAVERSAL ===
    float accum = 0.0f;
    float t = t_min;

    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);
    int iz = (int)metal::floor(src_z + t * dir_z + cz);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    int step_z = (dir_z >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float inv_dir_z = (metal::abs(dir_z) > eps) ? (1.0f / dir_z) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;
    float dt_z = (metal::abs(dir_z) > eps) ? metal::abs(inv_dir_z) : """ + _INF_STR + """;

    float txn = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float tyn = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;
    float tzn = (metal::abs(dir_z) > eps) ? ((float)(iz + (step_z > 0 ? 1 : 0)) - cz - src_z) * inv_dir_z : """ + _INF_STR + """;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                // Trilinear interpolation at midpoint
                float t_mid = t + seg_len * 0.5f;
                float mid_x = src_x + t_mid * dir_x + cx;
                float mid_y = src_y + t_mid * dir_y + cy;
                float mid_z = src_z + t_mid * dir_z + cz;

                int ix0 = (int)metal::floor(mid_x);
                int iy0 = (int)metal::floor(mid_y);
                int iz0 = (int)metal::floor(mid_z);
                float dx = mid_x - (float)ix0;
                float dy = mid_y - (float)iy0;
                float dz = mid_z - (float)iz0;

                ix0 = metal::max(0, metal::min(ix0, Nx - 2));
                iy0 = metal::max(0, metal::min(iy0, Ny - 2));
                iz0 = metal::max(0, metal::min(iz0, Nz - 2));

                float omdx = 1.0f - dx;
                float omdy = 1.0f - dy;
                float omdz = 1.0f - dz;

                // Volume layout: (Nx, Ny, Nz) = WHD permuted
                int base = ix0 * Ny * Nz + iy0 * Nz + iz0;
                int yz_stride = Ny * Nz;

                float val = vol[base]                         * omdx * omdy * omdz +
                            vol[base + yz_stride]             * dx   * omdy * omdz +
                            vol[base + Nz]                    * omdx * dy   * omdz +
                            vol[base + 1]                     * omdx * omdy * dz   +
                            vol[base + yz_stride + Nz]        * dx   * dy   * omdz +
                            vol[base + yz_stride + 1]         * dx   * omdy * dz   +
                            vol[base + Nz + 1]                * omdx * dy   * dz   +
                            vol[base + yz_stride + Nz + 1]    * dx   * dy   * dz;

                accum += val * seg_len;
            }
        }

        // 3D boundary crossing
        if (txn <= tyn && txn <= tzn) {
            t = txn;
            ix += step_x;
            txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = tyn;
            iy += step_y;
            tyn += dt_y;
        } else {
            t = tzn;
            iz += step_z;
            tzn += dt_z;
        }
    }

    sino[iview * n_u * n_v + iu * n_v + iv] = accum;
"""

cone_3d_forward_kernel = mx.fast.metal_kernel(
    name="cone_3d_forward",
    input_names=["vol", "src_pos", "det_center", "det_u_vec_arr", "det_v_vec_arr", "params", "fparams"],
    output_names=["sino"],
    source=_CONE_3D_FORWARD_SOURCE,
)

# ============================================================================
# 3D Cone Beam Backprojection Metal Kernel
# ============================================================================

_CONE_3D_BACKWARD_SOURCE = """
    // Thread position: (iview, iu, iv)
    uint iview = thread_position_in_grid.x;
    uint iu    = thread_position_in_grid.y;
    uint iv    = thread_position_in_grid.z;

    // Read scalar parameters
    int n_views = params[0];
    int n_u     = params[1];
    int n_v     = params[2];
    int Nx      = params[3];
    int Ny      = params[4];
    int Nz      = params[5];

    float du            = fparams[0];
    float dv            = fparams[1];
    float cx            = fparams[2];
    float cy            = fparams[3];
    float cz            = fparams[4];
    float voxel_spacing = fparams[5];

    if ((int)iview >= n_views || (int)iu >= n_u || (int)iv >= n_v) return;

    float eps = """ + _EPSILON_STR + """;

    float g = sino[iview * n_u * n_v + iu * n_v + iv];

    // === 3D CONE BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iview * 3 + 0] / voxel_spacing;
    float src_y = src_pos[iview * 3 + 1] / voxel_spacing;
    float src_z = src_pos[iview * 3 + 2] / voxel_spacing;

    float det_cx = det_center_arr[iview * 3 + 0] / voxel_spacing;
    float det_cy = det_center_arr[iview * 3 + 1] / voxel_spacing;
    float det_cz = det_center_arr[iview * 3 + 2] / voxel_spacing;

    float u_vec_x = det_u_vec_arr[iview * 3 + 0];
    float u_vec_y = det_u_vec_arr[iview * 3 + 1];
    float u_vec_z = det_u_vec_arr[iview * 3 + 2];

    float v_vec_x = det_v_vec_arr[iview * 3 + 0];
    float v_vec_y = det_v_vec_arr[iview * 3 + 1];
    float v_vec_z = det_v_vec_arr[iview * 3 + 2];

    float u_offset = ((float)iu - (float)n_u * 0.5f) * du / voxel_spacing;
    float v_offset = ((float)iv - (float)n_v * 0.5f) * dv / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x;
    float det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y;
    float det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z;

    // === 3D RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float dir_z = det_z - src_z;
    float length = metal::sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    if (length < eps) return;
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;
    dir_z *= inv_len;

    // === 3D RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;

    if (metal::abs(dir_x) > eps) {
        float tx1 = (-cx - src_x) / dir_x;
        float tx2 = ( cx - src_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (src_x < -cx || src_x > cx) { return; }

    if (metal::abs(dir_y) > eps) {
        float ty1 = (-cy - src_y) / dir_y;
        float ty2 = ( cy - src_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (src_y < -cy || src_y > cy) { return; }

    if (metal::abs(dir_z) > eps) {
        float tz1 = (-cz - src_z) / dir_z;
        float tz2 = ( cz - src_z) / dir_z;
        t_min = metal::max(t_min, metal::min(tz1, tz2));
        t_max = metal::min(t_max, metal::max(tz1, tz2));
    } else if (src_z < -cz || src_z > cz) { return; }

    if (t_min >= t_max) return;

    // === 3D SIDDON TRAVERSAL ===
    float t = t_min;
    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);
    int iz = (int)metal::floor(src_z + t * dir_z + cz);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    int step_z = (dir_z >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float inv_dir_z = (metal::abs(dir_z) > eps) ? (1.0f / dir_z) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;
    float dt_z = (metal::abs(dir_z) > eps) ? metal::abs(inv_dir_z) : """ + _INF_STR + """;

    float txn = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float tyn = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;
    float tzn = (metal::abs(dir_z) > eps) ? ((float)(iz + (step_z > 0 ? 1 : 0)) - cz - src_z) * inv_dir_z : """ + _INF_STR + """;

    int yz_stride = Ny * Nz;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                float t_mid = t + seg_len * 0.5f;
                float mid_x = src_x + t_mid * dir_x + cx;
                float mid_y = src_y + t_mid * dir_y + cy;
                float mid_z = src_z + t_mid * dir_z + cz;

                int ix0 = (int)metal::floor(mid_x);
                int iy0 = (int)metal::floor(mid_y);
                int iz0 = (int)metal::floor(mid_z);
                float dx = mid_x - (float)ix0;
                float dy = mid_y - (float)iy0;
                float dz = mid_z - (float)iz0;

                ix0 = metal::max(0, metal::min(ix0, Nx - 2));
                iy0 = metal::max(0, metal::min(iy0, Ny - 2));
                iz0 = metal::max(0, metal::min(iz0, Nz - 2));

                float omdx = 1.0f - dx;
                float omdy = 1.0f - dy;
                float omdz = 1.0f - dz;
                float cval = g * seg_len;

                int base = ix0 * yz_stride + iy0 * Nz + iz0;

                // Atomic backprojection with trilinear weights
                atomic_fetch_add_explicit(&grad_vol[base],                         cval * omdx * omdy * omdz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + yz_stride],             cval * dx   * omdy * omdz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + Nz],                    cval * omdx * dy   * omdz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + 1],                     cval * omdx * omdy * dz,   memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + yz_stride + Nz],        cval * dx   * dy   * omdz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + yz_stride + 1],         cval * dx   * omdy * dz,   memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + Nz + 1],                cval * omdx * dy   * dz,   memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + yz_stride + Nz + 1],    cval * dx   * dy   * dz,   memory_order_relaxed);
            }
        }

        if (txn <= tyn && txn <= tzn) {
            t = txn;
            ix += step_x;
            txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = tyn;
            iy += step_y;
            tyn += dt_y;
        } else {
            t = tzn;
            iz += step_z;
            tzn += dt_z;
        }
    }
"""

cone_3d_backward_kernel = mx.fast.metal_kernel(
    name="cone_3d_backward",
    input_names=["sino", "src_pos", "det_center_arr", "det_u_vec_arr", "det_v_vec_arr", "params", "fparams"],
    output_names=["grad_vol"],
    source=_CONE_3D_BACKWARD_SOURCE,
    atomic_outputs=True,
)
