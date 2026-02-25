"""Quick smoke test for diffct_mlx package."""

import mlx.core as mx
import diffct_mlx

print("Version:", diffct_mlx.__version__)

# ---- 2D Parallel Beam Test ----
print("\n=== 2D Parallel Beam Forward Projection ===")
Nx, Ny = 64, 64
n_angles, n_det = 90, 92
pixel_size = 1.0

image = mx.ones((Nx, Ny), dtype=mx.float32)

ray_dir, det_origin, det_u_vec = diffct_mlx.circular_trajectory_2d_parallel(
    n_angles, detector_distance=0.0
)

sino = diffct_mlx.parallel_forward(
    image, ray_dir, det_origin, det_u_vec,
    n_det, pixel_size, pixel_size
)
mx.eval(sino)
print("  sino shape:", sino.shape, "dtype:", sino.dtype)
print("  sino min:", float(sino.min()), "max:", float(sino.max()), "mean:", float(sino.mean()))

# ---- 2D Parallel Beam Backward ----
print("\n=== 2D Parallel Beam Backward Projection ===")
bp = diffct_mlx.parallel_backward(
    sino, ray_dir, det_origin, det_u_vec,
    detector_spacing=pixel_size, H=Nx, W=Ny, voxel_spacing=pixel_size
)
mx.eval(bp)
print("  bp shape:", bp.shape, "dtype:", bp.dtype)
print("  bp min:", float(bp.min()), "max:", float(bp.max()), "mean:", float(bp.mean()))

# ---- 2D Fan Beam Test ----
print("\n=== 2D Fan Beam Forward Projection ===")
sid, sdd = 500.0, 1000.0

src_pos, det_center, det_u = diffct_mlx.circular_trajectory_2d_fan(
    n_angles, sid, sdd
)

sino_fan = diffct_mlx.fan_forward(
    image, src_pos, det_center, det_u,
    n_det, pixel_size, pixel_size
)
mx.eval(sino_fan)
print("  sino shape:", sino_fan.shape, "dtype:", sino_fan.dtype)
print("  sino min:", float(sino_fan.min()), "max:", float(sino_fan.max()), "mean:", float(sino_fan.mean()))

# ---- 2D Fan Beam Backward ----
print("\n=== 2D Fan Beam Backward Projection ===")
bp_fan = diffct_mlx.fan_backward(
    sino_fan, src_pos, det_center, det_u,
    detector_spacing=pixel_size, H=Nx, W=Ny, voxel_spacing=pixel_size
)
mx.eval(bp_fan)
print("  bp shape:", bp_fan.shape, "dtype:", bp_fan.dtype)
print("  bp min:", float(bp_fan.min()), "max:", float(bp_fan.max()), "mean:", float(bp_fan.mean()))

# ---- 3D Cone Beam Test ----
print("\n=== 3D Cone Beam Forward Projection ===")
Nz = 32
n_det_u, n_det_v = 64, 32
n_views_3d = 60

volume = mx.ones((Nz, Ny, Nx), dtype=mx.float32)

src_3d, det_c_3d, det_u_3d, det_v_3d = diffct_mlx.circular_trajectory_3d(
    n_views_3d, sid, sdd
)

sino_cone = diffct_mlx.cone_forward(
    volume, src_3d, det_c_3d, det_u_3d, det_v_3d,
    n_det_u, n_det_v, pixel_size, pixel_size, pixel_size
)
mx.eval(sino_cone)
print("  sino shape:", sino_cone.shape, "dtype:", sino_cone.dtype)
print("  sino min:", float(sino_cone.min()), "max:", float(sino_cone.max()), "mean:", float(sino_cone.mean()))

# ---- 3D Cone Beam Backward ----
print("\n=== 3D Cone Beam Backward Projection ===")
bp_cone = diffct_mlx.cone_backward(
    sino_cone, src_3d, det_c_3d, det_u_3d, det_v_3d,
    D=Nz, H=Ny, W=Nx, du=pixel_size, dv=pixel_size, voxel_spacing=pixel_size
)
mx.eval(bp_cone)
print("  bp shape:", bp_cone.shape, "dtype:", bp_cone.dtype)
print("  bp min:", float(bp_cone.min()), "max:", float(bp_cone.max()), "mean:", float(bp_cone.mean()))

print("\n=== All tests passed! ===")
