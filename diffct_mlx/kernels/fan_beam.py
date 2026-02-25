"""Metal kernels for 2D fan beam projections.

This module contains Metal kernel source strings implementing the Siddon
ray-tracing method for 2D fan beam forward projection and backprojection,
optimized for Apple Silicon GPUs.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants for Metal kernels
# ---------------------------------------------------------------------------
_EPSILON_STR = "1e-6f"
_INF_STR = "1e30f"

# ============================================================================
# 2D Fan Beam Forward Projection Metal Kernel
# ============================================================================

_FAN_2D_FORWARD_SOURCE = """
    // Thread position: (iang, idet)
    uint iang = thread_position_in_grid.x;
    uint idet = thread_position_in_grid.y;

    // Read scalar parameters
    int n_ang = params[0];
    int n_det = params[1];
    int Nx    = params[2];
    int Ny    = params[3];

    float det_spacing   = fparams[0];
    float cx            = fparams[1];
    float cy            = fparams[2];
    float voxel_spacing = fparams[3];

    if ((int)iang >= n_ang || (int)idet >= n_det) return;

    float eps = """ + _EPSILON_STR + """;

    // === 2D FAN BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
    float src_y = src_pos[iang * 2 + 1] / voxel_spacing;

    float det_cx = det_center[iang * 2 + 0] / voxel_spacing;
    float det_cy = det_center[iang * 2 + 1] / voxel_spacing;

    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_offset = ((float)idet - (float)n_det * 0.5f) * det_spacing / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x;
    float det_y = det_cy + u_offset * u_vec_y;

    // === RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float length = metal::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (length < eps) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;

    // === RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;

    if (metal::abs(dir_x) > eps) {
        float tx1 = (-cx - src_x) / dir_x;
        float tx2 = ( cx - src_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (src_x < -cx || src_x > cx) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    if (metal::abs(dir_y) > eps) {
        float ty1 = (-cy - src_y) / dir_y;
        float ty2 = ( cy - src_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (src_y < -cy || src_y > cy) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    if (t_min >= t_max) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    // === SIDDON TRAVERSAL ===
    float accum = 0.0f;
    float t = t_min;

    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;

    float tx = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float ty = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            float t_next = metal::min(metal::min(tx, ty), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                float t_mid = t + seg_len * 0.5f;
                float mid_x = src_x + t_mid * dir_x + cx;
                float mid_y = src_y + t_mid * dir_y + cy;

                int ix0 = (int)metal::floor(mid_x);
                int iy0 = (int)metal::floor(mid_y);
                float dx = mid_x - (float)ix0;
                float dy = mid_y - (float)iy0;

                ix0 = metal::max(0, metal::min(ix0, Nx - 2));
                iy0 = metal::max(0, metal::min(iy0, Ny - 2));

                float omdx = 1.0f - dx;
                float omdy = 1.0f - dy;

                float v00 = image[iy0 * Nx + ix0];
                float v10 = image[iy0 * Nx + ix0 + 1];
                float v01 = image[(iy0 + 1) * Nx + ix0];
                float v11 = image[(iy0 + 1) * Nx + ix0 + 1];

                float val = (v00 * omdx + v10 * dx) * omdy +
                            (v01 * omdx + v11 * dx) * dy;
                accum += val * seg_len;
            }
        }

        if (tx <= ty) {
            t = tx;
            ix += step_x;
            tx += dt_x;
        } else {
            t = ty;
            iy += step_y;
            ty += dt_y;
        }
    }

    sino[iang * n_det + idet] = accum;
"""

fan_2d_forward_kernel = mx.fast.metal_kernel(
    name="fan_2d_forward",
    input_names=["image", "src_pos", "det_center", "det_u_vec", "params", "fparams"],
    output_names=["sino"],
    source=_FAN_2D_FORWARD_SOURCE,
)

# ============================================================================
# 2D Fan Beam Backprojection Metal Kernel
# ============================================================================

_FAN_2D_BACKWARD_SOURCE = """
    // Thread position: (iang, idet)
    uint iang = thread_position_in_grid.x;
    uint idet = thread_position_in_grid.y;

    // Read scalar parameters
    int n_ang = params[0];
    int n_det = params[1];
    int Nx    = params[2];
    int Ny    = params[3];

    float det_spacing   = fparams[0];
    float cx            = fparams[1];
    float cy            = fparams[2];
    float voxel_spacing = fparams[3];

    if ((int)iang >= n_ang || (int)idet >= n_det) return;

    float eps = """ + _EPSILON_STR + """;

    float val = sino[iang * n_det + idet];

    // === 2D FAN BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
    float src_y = src_pos[iang * 2 + 1] / voxel_spacing;

    float det_cx = det_center_arr[iang * 2 + 0] / voxel_spacing;
    float det_cy = det_center_arr[iang * 2 + 1] / voxel_spacing;

    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_offset = ((float)idet - (float)n_det * 0.5f) * det_spacing / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x;
    float det_y = det_cy + u_offset * u_vec_y;

    // === RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float length = metal::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (length < eps) return;
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;

    // === RAY-VOLUME INTERSECTION ===
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

    if (t_min >= t_max) return;

    // === SIDDON TRAVERSAL ===
    float t = t_min;
    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;

    float tx = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float ty = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            float t_next = metal::min(metal::min(tx, ty), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                float t_mid = t + seg_len * 0.5f;
                float mid_x = src_x + t_mid * dir_x + cx;
                float mid_y = src_y + t_mid * dir_y + cy;

                int ix0 = (int)metal::floor(mid_x);
                int iy0 = (int)metal::floor(mid_y);
                float dx = mid_x - (float)ix0;
                float dy = mid_y - (float)iy0;

                ix0 = metal::max(0, metal::min(ix0, Nx - 2));
                iy0 = metal::max(0, metal::min(iy0, Ny - 2));

                float omdx = 1.0f - dx;
                float omdy = 1.0f - dy;
                float cval = val * seg_len;

                atomic_fetch_add_explicit(&grad_image[iy0 * Nx + ix0],           cval * omdx * omdy, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_image[iy0 * Nx + ix0 + 1],       cval * dx   * omdy, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_image[(iy0 + 1) * Nx + ix0],     cval * omdx * dy,   memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_image[(iy0 + 1) * Nx + ix0 + 1], cval * dx   * dy,   memory_order_relaxed);
            }
        }

        if (tx <= ty) {
            t = tx;
            ix += step_x;
            tx += dt_x;
        } else {
            t = ty;
            iy += step_y;
            ty += dt_y;
        }
    }
"""

fan_2d_backward_kernel = mx.fast.metal_kernel(
    name="fan_2d_backward",
    input_names=["sino", "src_pos", "det_center_arr", "det_u_vec", "params", "fparams"],
    output_names=["grad_image"],
    source=_FAN_2D_BACKWARD_SOURCE,
    atomic_outputs=True,
)
