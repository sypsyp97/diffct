import math
import numpy as np
import torch
from numba import cuda

# ---------------------------------------------------------------------------
# Global settings & helpers
# ---------------------------------------------------------------------------

_DTYPE             = np.float32            # change to np.float64 if desired
_TPB_2D            = (16, 16)
_TPB_3D            = (8,  8,  8)
_FASTMATH_DECORATOR = cuda.jit(fastmath=True)


def _trig_tables(angles: np.ndarray, dtype=_DTYPE):
    """Return device arrays (d_cos, d_sin) for all projection angles."""
    cos_host = np.cos(angles).astype(dtype)
    sin_host = np.sin(angles).astype(dtype)
    return cuda.to_device(cos_host), cuda.to_device(sin_host)


def _grid_2d(n1, n2, tpb=_TPB_2D):
    return (math.ceil(n1 / tpb[0]), math.ceil(n2 / tpb[1])), tpb


def _grid_3d(n1, n2, n3, tpb=_TPB_3D):
    return (
        math.ceil(n1 / tpb[0]),
        math.ceil(n2 / tpb[1]),
        math.ceil(n3 / tpb[2]),
    ), tpb


# ############################################################################
# Parallel beam – differentiable projector
# ############################################################################
class ParallelProjectorFunction(torch.autograd.Function):

    # -------------------------- CUDA kernels --------------------------------
    @_FASTMATH_DECORATOR
    def _forward_kernel(
        d_image,
        Nx, Ny, diag,
        d_sino,
        n_ang, n_det,
        det_spacing,
        d_cos, d_sin,
        step, cx, cy
    ):
        iang, idet = cuda.grid(2)
        if iang >= n_ang or idet >= n_det:
            return

        cos_a = d_cos[iang]
        sin_a = d_sin[iang]
        u     = (idet - (n_det - 1) * 0.5) * det_spacing

        t     = -diag
        accum = 0.0
        while t < diag:
            x  = u * (-sin_a) + t * cos_a
            y  = u * ( cos_a) + t * sin_a
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx = ix - ix0
                dy = iy - iy0
                accum += (
                    d_image[ix0,     iy0]     * (1 - dx) * (1 - dy) +
                    d_image[ix0 + 1, iy0]     * dx       * (1 - dy) +
                    d_image[ix0,     iy0 + 1] * (1 - dx) * dy       +
                    d_image[ix0 + 1, iy0 + 1] * dx       * dy
                ) * step
            t += step
        d_sino[iang, idet] = accum

    @_FASTMATH_DECORATOR
    def _backward_kernel(
        d_grad_sino,
        n_ang, n_det,
        Nx, Ny, diag,
        d_img_grad,
        det_spacing,
        d_cos, d_sin,
        step, cx, cy
    ):
        iang, idet = cuda.grid(2)
        if iang >= n_ang or idet >= n_det:
            return

        g     = d_grad_sino[iang, idet]
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]
        u     = (idet - (n_det - 1) * 0.5) * det_spacing

        t = -diag
        while t < diag:
            x  = u * (-sin_a) + t * cos_a
            y  = u * ( cos_a) + t * sin_a
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx = ix - ix0
                dy = iy - iy0
                cval = g * step
                cuda.atomic.add(d_img_grad, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_img_grad, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
                cuda.atomic.add(d_img_grad, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
                cuda.atomic.add(d_img_grad, (ix0 + 1, iy0 + 1), cval * dx       * dy)
            t += step

    # --------------------------- forward ------------------------------------
    @staticmethod
    def forward(ctx, image, angles,
                num_detectors, detector_spacing=1.0, step_size=0.5):
        device  = image.device
        image_np   = image.detach().cpu().numpy().astype(_DTYPE, copy=False)
        angles_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        Nx, Ny    = image_np.shape
        n_angles  = angles_np.shape[0]
        diag      = _DTYPE(math.sqrt(Nx * Nx + Ny * Ny))

        d_image   = cuda.to_device(image_np)
        d_cos, d_sin = _trig_tables(angles_np, _DTYPE)
        d_sino    = cuda.device_array((n_angles, num_detectors), dtype=_DTYPE)

        (grid, tpb) = _grid_2d(n_angles, num_detectors)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)

        ParallelProjectorFunction._forward_kernel[grid, tpb](
            d_image,
            Nx, Ny, diag,
            d_sino,
            n_angles, num_detectors,
            _DTYPE(detector_spacing),
            d_cos, d_sin,
            _DTYPE(step_size), cx, cy
        )

        sinogram = torch.as_tensor(d_sino, device=device)
        ctx.save_for_backward(image, angles)
        ctx.intermediate = (num_detectors, detector_spacing, step_size, Nx, Ny)
        return sinogram

    # --------------------------- backward -----------------------------------
    @staticmethod
    def backward(ctx, grad_sinogram):
        image, angles = ctx.saved_tensors
        (n_det, det_spacing, step_size, Nx, Ny) = ctx.intermediate
        device = image.device

        grad_np = grad_sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        n_angles = ang_np.shape[0]
        diag     = _DTYPE(math.sqrt(Nx * Nx + Ny * Ny))

        d_grad_sino = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_img_grad   = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

        (grid, tpb) = _grid_2d(n_angles, n_det)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)

        ParallelProjectorFunction._backward_kernel[grid, tpb](
            d_grad_sino,
            n_angles, n_det,
            Nx, Ny, diag,
            d_img_grad,
            _DTYPE(det_spacing),
            d_cos, d_sin,
            _DTYPE(step_size), cx, cy
        )

        grad_image = torch.as_tensor(d_img_grad, device=device)
        return grad_image, None, None, None, None


# ############################################################################
# Parallel beam – differentiable back‑projector
# ############################################################################
class ParallelBackprojectorFunction(torch.autograd.Function):

    # -------------------------- CUDA kernels --------------------------------
    @_FASTMATH_DECORATOR
    def _forward_kernel(
        d_sino,
        n_ang, n_det,
        Nx, Ny, diag,
        d_reco,
        det_spacing,
        d_cos, d_sin,
        step, cx, cy
    ):
        iang, idet = cuda.grid(2)
        if iang >= n_ang or idet >= n_det:
            return
        val   = d_sino[iang, idet]
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]
        u     = (idet - (n_det - 1) * 0.5) * det_spacing

        t = -diag
        while t < diag:
            x  = u * (-sin_a) + t * cos_a
            y  = u * ( cos_a) + t * sin_a
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx = ix - ix0
                dy = iy - iy0
                cval = val * step
                cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1), cval * dx       * dy)
            t += step

    @_FASTMATH_DECORATOR
    def _backward_kernel(
        d_grad_out,
        n_ang, n_det,
        Nx, Ny, diag,
        d_sino_grad,
        det_spacing,
        d_cos, d_sin,
        step, cx, cy
    ):
        iang, idet = cuda.grid(2)
        if iang >= n_ang or idet >= n_det:
            return

        cos_a = d_cos[iang]
        sin_a = d_sin[iang]
        u     = (idet - (n_det - 1) * 0.5) * det_spacing

        t = -diag
        while t < diag:
            x  = u * (-sin_a) + t * cos_a
            y  = u * ( cos_a) + t * sin_a
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx = ix - ix0
                dy = iy - iy0
                cval = (
                    d_grad_out[ix0,     iy0]     * (1 - dx) * (1 - dy) +
                    d_grad_out[ix0 + 1, iy0]     * dx       * (1 - dy) +
                    d_grad_out[ix0,     iy0 + 1] * (1 - dx) * dy       +
                    d_grad_out[ix0 + 1, iy0 + 1] * dx       * dy
                ) * step
                cuda.atomic.add(d_sino_grad, (iang, idet), cval)
            t += step

    # --------------------------- forward ------------------------------------
    @staticmethod
    def forward(ctx, sinogram, angles,
                detector_spacing=1.0, step_size=0.5,
                Nx=128, Ny=128):
        device   = sinogram.device
        sino_np  = sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        angles_np= angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        n_ang, n_det = sino_np.shape
        diag   = _DTYPE(math.sqrt(Nx * Nx + Ny * Ny))

        d_sino = cuda.to_device(sino_np)
        d_cos, d_sin = _trig_tables(angles_np, _DTYPE)
        d_reco = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

        (grid, tpb) = _grid_2d(n_ang, n_det)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)

        ParallelBackprojectorFunction._forward_kernel[grid, tpb](
            d_sino,
            n_ang, n_det,
            Nx, Ny, diag,
            d_reco,
            _DTYPE(detector_spacing),
            d_cos, d_sin,
            _DTYPE(step_size), cx, cy
        )

        reco = torch.as_tensor(d_reco, device=device)
        ctx.save_for_backward(sinogram, angles)
        ctx.intermediate = (Nx, Ny, detector_spacing, step_size)
        return reco

    # --------------------------- backward -----------------------------------
    @staticmethod
    def backward(ctx, grad_output):
        sinogram, angles = ctx.saved_tensors
        Nx, Ny, det_spacing, step_size = ctx.intermediate
        device = sinogram.device

        grad_np  = grad_output.detach().cpu().numpy().astype(_DTYPE, copy=False)
        angles_np= angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_ang, n_det = sinogram.shape
        diag   = _DTYPE(math.sqrt(Nx * Nx + Ny * Ny))

        d_grad_out = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(angles_np, _DTYPE)
        d_sino_grad = cuda.to_device(np.zeros((n_ang, n_det), dtype=_DTYPE))

        (grid, tpb) = _grid_2d(n_ang, n_det)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)

        ParallelBackprojectorFunction._backward_kernel[grid, tpb](
            d_grad_out,
            n_ang, n_det,
            Nx, Ny, diag,
            d_sino_grad,
            _DTYPE(det_spacing),
            d_cos, d_sin,
            _DTYPE(step_size), cx, cy
        )

        grad_sino = torch.as_tensor(d_sino_grad, device=device)
        return grad_sino, None, None, None, None, None


# ############################################################################
# Fan‑beam projector / back‑projector
# ############################################################################
class FanProjectorFunction(torch.autograd.Function):

    @_FASTMATH_DECORATOR
    def _forward_kernel(
        d_image,
        Nx, Ny,
        d_sino,
        n_ang, n_det,
        det_spacing,
        d_cos, d_sin,
        step,
        src_dist, iso_dist,
        cx, cy
    ):
        iang, idet = cuda.grid(2)
        if iang >= n_ang or idet >= n_det:
            return
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]
        u     = (idet - (n_det - 1) * 0.5) * det_spacing

        sx = -iso_dist * sin_a
        sy =  iso_dist * cos_a
        dx = (src_dist - iso_dist) * sin_a + u * cos_a
        dy = -(src_dist - iso_dist) * cos_a + u * sin_a

        rx = dx - sx
        ry = dy - sy
        length   = math.sqrt(rx * rx + ry * ry)
        inv_len  = 1.0 / length
        rx *= inv_len
        ry *= inv_len

        accum = 0.0
        t = 0.0
        while t < length:
            x  = sx + t * rx
            y  = sy + t * ry
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx2 = ix - ix0
                dy2 = iy - iy0
                accum += (
                    d_image[ix0,     iy0]     * (1 - dx2) * (1 - dy2) +
                    d_image[ix0 + 1, iy0]     * dx2       * (1 - dy2) +
                    d_image[ix0,     iy0 + 1] * (1 - dx2) * dy2       +
                    d_image[ix0 + 1, iy0 + 1] * dx2       * dy2
                ) * step
            t += step
        d_sino[iang, idet] = accum

    @_FASTMATH_DECORATOR
    def _backward_kernel(
        d_grad_sino,
        n_ang, n_det,
        Nx, Ny,
        d_img_grad,
        det_spacing,
        d_cos, d_sin,
        step,
        src_dist, iso_dist,
        cx, cy
    ):
        iang, idet = cuda.grid(2)
        if iang >= n_ang or idet >= n_det:
            return
        g     = d_grad_sino[iang, idet]
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]
        u     = (idet - (n_det - 1) * 0.5) * det_spacing

        sx = -iso_dist * sin_a
        sy =  iso_dist * cos_a
        dx = (src_dist - iso_dist) * sin_a + u * cos_a
        dy = -(src_dist - iso_dist) * cos_a + u * sin_a
        rx = dx - sx
        ry = dy - sy
        length  = math.sqrt(rx * rx + ry * ry)
        inv_len = 1.0 / length
        rx *= inv_len
        ry *= inv_len

        t = 0.0
        while t < length:
            x  = sx + t * rx
            y  = sy + t * ry
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx2 = ix - ix0
                dy2 = iy - iy0
                cval = g * step
                cuda.atomic.add(d_img_grad, (ix0,     iy0),     cval * (1 - dx2) * (1 - dy2))
                cuda.atomic.add(d_img_grad, (ix0 + 1, iy0),     cval * dx2       * (1 - dy2))
                cuda.atomic.add(d_img_grad, (ix0,     iy0 + 1), cval * (1 - dx2) * dy2)
                cuda.atomic.add(d_img_grad, (ix0 + 1, iy0 + 1), cval * dx2       * dy2)
            t += step

    # --------------------------- forward ------------------------------------
    @staticmethod
    def forward(ctx, image, angles,
                num_detectors, detector_spacing,
                step_size, source_distance, isocenter_distance):
        device = image.device
        img_np   = image.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np   = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        Nx, Ny  = img_np.shape
        n_ang   = ang_np.shape[0]

        d_image = cuda.to_device(img_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_sino  = cuda.device_array((n_ang, num_detectors), dtype=_DTYPE)

        (grid, tpb) = _grid_2d(n_ang, num_detectors)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)

        FanProjectorFunction._forward_kernel[grid, tpb](
            d_image,
            Nx, Ny,
            d_sino,
            n_ang, num_detectors,
            _DTYPE(detector_spacing),
            d_cos, d_sin,
            _DTYPE(step_size),
            _DTYPE(source_distance), _DTYPE(isocenter_distance),
            cx, cy
        )

        sino = torch.as_tensor(d_sino, device=device)
        ctx.save_for_backward(image, angles)
        ctx.intermediate = (num_detectors, detector_spacing, step_size,
                            Nx, Ny, source_distance, isocenter_distance)
        return sino

    # --------------------------- backward -----------------------------------
    @staticmethod
    def backward(ctx, grad_sinogram):
        image, angles = ctx.saved_tensors
        (n_det, det_spacing, step_size,
         Nx, Ny, src_dist, iso_dist) = ctx.intermediate
        device  = image.device

        grad_np = grad_sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_ang   = ang_np.shape[0]

        d_grad_sino = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_img_grad   = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

        (grid, tpb) = _grid_2d(n_ang, n_det)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)

        FanProjectorFunction._backward_kernel[grid, tpb](
            d_grad_sino,
            n_ang, n_det,
            Nx, Ny,
            d_img_grad,
            _DTYPE(det_spacing),
            d_cos, d_sin,
            _DTYPE(step_size),
            _DTYPE(src_dist), _DTYPE(iso_dist),
            cx, cy
        )
        grad_img = torch.as_tensor(d_img_grad, device=device)
        return grad_img, None, None, None, None, None, None


# ############################################################################
# Fan‑beam back‑projector
# ############################################################################
class FanBackprojectorFunction(torch.autograd.Function):
    # Kernels are analogous – only the trig optimisation changed
    @_FASTMATH_DECORATOR
    def _forward_kernel(
        d_sino,
        n_ang, n_det,
        Nx, Ny,
        d_reco,
        det_spacing,
        d_cos, d_sin,
        step,
        src_dist, iso_dist,
        cx, cy
    ):
        iang, idet = cuda.grid(2)
        if iang >= n_ang or idet >= n_det:
            return
        val   = d_sino[iang, idet]
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]
        u     = (idet - (n_det - 1) * 0.5) * det_spacing

        sx = -iso_dist * sin_a
        sy =  iso_dist * cos_a
        dx = (src_dist - iso_dist) * sin_a + u * cos_a
        dy = -(src_dist - iso_dist) * cos_a + u * sin_a
        rx = dx - sx
        ry = dy - sy
        length  = math.sqrt(rx * rx + ry * ry)
        inv_len = 1.0 / length
        rx *= inv_len
        ry *= inv_len

        t = 0.0
        while t < length:
            x  = sx + t * rx
            y  = sy + t * ry
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx2 = ix - ix0
                dy2 = iy - iy0
                cval = val * step
                cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx2) * (1 - dy2))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx2       * (1 - dy2))
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1), cval * (1 - dx2) * dy2)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1), cval * dx2       * dy2)
            t += step

    @_FASTMATH_DECORATOR
    def _backward_kernel(
        d_grad_out,
        n_ang, n_det,
        Nx, Ny,
        d_sino_grad,
        det_spacing,
        d_cos, d_sin,
        step,
        src_dist, iso_dist,
        cx, cy
    ):
        iang, idet = cuda.grid(2)
        if iang >= n_ang or idet >= n_det:
            return
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]
        u     = (idet - (n_det - 1) * 0.5) * det_spacing

        sx = -iso_dist * sin_a
        sy =  iso_dist * cos_a
        dx = (src_dist - iso_dist) * sin_a + u * cos_a
        dy = -(src_dist - iso_dist) * cos_a + u * sin_a
        rx = dx - sx
        ry = dy - sy
        length  = math.sqrt(rx * rx + ry * ry)
        inv_len = 1.0 / length
        rx *= inv_len
        ry *= inv_len

        t = 0.0
        while t < length:
            x  = sx + t * rx
            y  = sy + t * ry
            ix = x + cx
            iy = y + cy
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
                dx2 = ix - ix0
                dy2 = iy - iy0
                cval = (
                    d_grad_out[ix0,     iy0]     * (1 - dx2) * (1 - dy2) +
                    d_grad_out[ix0 + 1, iy0]     * dx2       * (1 - dy2) +
                    d_grad_out[ix0,     iy0 + 1] * (1 - dx2) * dy2       +
                    d_grad_out[ix0 + 1, iy0 + 1] * dx2       * dy2
                ) * step
                cuda.atomic.add(d_sino_grad, (iang, idet), cval)
            t += step

    # ---------------- forward -----------------
    @staticmethod
    def forward(ctx, sinogram, angles,
                detector_spacing, step_size,
                Nx, Ny,
                source_distance, isocenter_distance):
        device = sinogram.device
        sino_np = sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_ang, n_det = sino_np.shape

        d_sino = cuda.to_device(sino_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_reco = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

        (grid, tpb) = _grid_2d(n_ang, n_det)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)

        FanBackprojectorFunction._forward_kernel[grid, tpb](
            d_sino,
            n_ang, n_det,
            Nx, Ny,
            d_reco,
            _DTYPE(detector_spacing),
            d_cos, d_sin,
            _DTYPE(step_size),
            _DTYPE(source_distance), _DTYPE(isocenter_distance),
            cx, cy
        )

        image = torch.as_tensor(d_reco, device=device)
        ctx.save_for_backward(sinogram, angles)
        ctx.intermediate = (Nx, Ny, detector_spacing, step_size,
                            source_distance, isocenter_distance)
        return image

    # -------------- backward ------------------
    @staticmethod
    def backward(ctx, grad_output):
        sinogram, angles = ctx.saved_tensors
        (Nx, Ny, det_spacing, step_size,
         src_dist, iso_dist) = ctx.intermediate
        device = sinogram.device

        grad_np = grad_output.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_ang, n_det = sinogram.shape

        d_grad_out = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_sino_grad = cuda.to_device(np.zeros((n_ang, n_det), dtype=_DTYPE))

        (grid, tpb) = _grid_2d(n_ang, n_det)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)

        FanBackprojectorFunction._backward_kernel[grid, tpb](
            d_grad_out,
            n_ang, n_det,
            Nx, Ny,
            d_sino_grad,
            _DTYPE(det_spacing),
            d_cos, d_sin,
            _DTYPE(step_size),
            _DTYPE(src_dist), _DTYPE(iso_dist),
            cx, cy
        )
        grad_sino = torch.as_tensor(d_sino_grad, device=device)
        return grad_sino, None, None, None, None, None, None, None


# ############################################################################
# Cone‑beam: differentiable forward projector
# ############################################################################
class ConeProjectorFunction(torch.autograd.Function):
    @_FASTMATH_DECORATOR
    def _forward_kernel(
        d_vol,
        Nx, Ny, Nz,
        d_sino,
        n_views, n_u, n_v,
        du, dv,
        d_cos, d_sin,
        step,
        src_dist, iso_dist,
        cx, cy, cz
    ):
        iview, iu, iv = cuda.grid(3)
        if iview >= n_views or iu >= n_u or iv >= n_v:
            return

        cos_a = d_cos[iview]
        sin_a = d_sin[iview]
        u     = (iu - (n_u - 1) * 0.5) * du
        v     = (iv - (n_v - 1) * 0.5) * dv

        sx = -iso_dist * sin_a
        sy =  iso_dist * cos_a
        sz = 0.0

        dx = (src_dist - iso_dist) * sin_a + u * cos_a
        dy = -(src_dist - iso_dist) * cos_a + u * sin_a
        dz = v

        rx = dx - sx
        ry = dy - sy
        rz = dz - sz
        length   = math.sqrt(rx * rx + ry * ry + rz * rz)
        inv_len  = 1.0 / length
        rx *= inv_len
        ry *= inv_len
        rz *= inv_len

        accum = 0.0
        t = 0.0
        while t < length:
            x  = sx + t * rx
            y  = sy + t * ry
            z  = sz + t * rz
            ix = x + cx
            iy = y + cy
            iz = z + cz
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            iz0 = int(math.floor(iz))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1 and 0 <= iz0 < Nz - 1:
                dx2 = ix - ix0
                dy2 = iy - iy0
                dz2 = iz - iz0
                accum += (
                    d_vol[ix0,     iy0,     iz0    ] * (1 - dx2) * (1 - dy2) * (1 - dz2) +
                    d_vol[ix0 + 1, iy0,     iz0    ] * dx2       * (1 - dy2) * (1 - dz2) +
                    d_vol[ix0,     iy0 + 1, iz0    ] * (1 - dx2) * dy2       * (1 - dz2) +
                    d_vol[ix0,     iy0,     iz0 + 1] * (1 - dx2) * (1 - dy2) * dz2       +
                    d_vol[ix0 + 1, iy0 + 1, iz0    ] * dx2       * dy2       * (1 - dz2) +
                    d_vol[ix0 + 1, iy0,     iz0 + 1] * dx2       * (1 - dy2) * dz2       +
                    d_vol[ix0,     iy0 + 1, iz0 + 1] * (1 - dx2) * dy2       * dz2       +
                    d_vol[ix0 + 1, iy0 + 1, iz0 + 1] * dx2       * dy2       * dz2
                ) * step
            t += step
        d_sino[iview, iu, iv] = accum

    @_FASTMATH_DECORATOR
    def _backward_kernel(
        d_grad_sino,
        n_views, n_u, n_v,
        Nx, Ny, Nz,
        d_vol_grad,
        du, dv,
        d_cos, d_sin,
        step,
        src_dist, iso_dist,
        cx, cy, cz
    ):
        iview, iu, iv = cuda.grid(3)
        if iview >= n_views or iu >= n_u or iv >= n_v:
            return

        g     = d_grad_sino[iview, iu, iv]
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]
        u     = (iu - (n_u - 1) * 0.5) * du
        v     = (iv - (n_v - 1) * 0.5) * dv

        sx = -iso_dist * sin_a
        sy =  iso_dist * cos_a
        sz = 0.0

        dx = (src_dist - iso_dist) * sin_a + u * cos_a
        dy = -(src_dist - iso_dist) * cos_a + u * sin_a
        dz = v

        rx = dx - sx
        ry = dy - sy
        rz = dz - sz
        length  = math.sqrt(rx * rx + ry * ry + rz * rz)
        inv_len = 1.0 / length
        rx *= inv_len
        ry *= inv_len
        rz *= inv_len

        t = 0.0
        while t < length:
            x  = sx + t * rx
            y  = sy + t * ry
            z  = sz + t * rz
            ix = x + cx
            iy = y + cy
            iz = z + cz
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            iz0 = int(math.floor(iz))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1 and 0 <= iz0 < Nz - 1:
                dx2 = ix - ix0
                dy2 = iy - iy0
                dz2 = iz - iz0
                cval = g * step
                cuda.atomic.add(d_vol_grad, (ix0,     iy0,     iz0    ), cval * (1 - dx2) * (1 - dy2) * (1 - dz2))
                cuda.atomic.add(d_vol_grad, (ix0 + 1, iy0,     iz0    ), cval * dx2       * (1 - dy2) * (1 - dz2))
                cuda.atomic.add(d_vol_grad, (ix0,     iy0 + 1, iz0    ), cval * (1 - dx2) * dy2       * (1 - dz2))
                cuda.atomic.add(d_vol_grad, (ix0,     iy0,     iz0 + 1), cval * (1 - dx2) * (1 - dy2) * dz2)
                cuda.atomic.add(d_vol_grad, (ix0 + 1, iy0 + 1, iz0    ), cval * dx2       * dy2       * (1 - dz2))
                cuda.atomic.add(d_vol_grad, (ix0 + 1, iy0,     iz0 + 1), cval * dx2       * (1 - dy2) * dz2)
                cuda.atomic.add(d_vol_grad, (ix0,     iy0 + 1, iz0 + 1), cval * (1 - dx2) * dy2       * dz2)
                cuda.atomic.add(d_vol_grad, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx2       * dy2       * dz2)
            t += step

    # --------------------------- forward ------------------------------------
    @staticmethod
    def forward(ctx, volume, angles,
                Nx, Ny, Nz,
                det_u, det_v, du, dv,
                step_size,
                source_distance, isocenter_distance):
        device  = volume.device
        vol_np  = volume.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        n_views = ang_np.shape[0]

        d_vol  = cuda.to_device(vol_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_sino = cuda.device_array((n_views, det_u, det_v), dtype=_DTYPE)

        (grid, tpb) = _grid_3d(n_views, det_u, det_v)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)
        cz = _DTYPE((Nz - 1) * 0.5)

        ConeProjectorFunction._forward_kernel[grid, tpb](
            d_vol,
            Nx, Ny, Nz,
            d_sino,
            n_views, det_u, det_v,
            _DTYPE(du), _DTYPE(dv),
            d_cos, d_sin,
            _DTYPE(step_size),
            _DTYPE(source_distance), _DTYPE(isocenter_distance),
            cx, cy, cz
        )

        sino = torch.as_tensor(d_sino, device=device)
        ctx.save_for_backward(volume, angles)
        ctx.intermediate = (Nx, Ny, Nz, det_u, det_v, du, dv,
                            step_size, source_distance, isocenter_distance)
        return sino

    # --------------------------- backward -----------------------------------
    @staticmethod
    def backward(ctx, grad_sinogram):
        volume, angles = ctx.saved_tensors
        (Nx, Ny, Nz, det_u, det_v, du, dv,
         step_size, src_dist, iso_dist) = ctx.intermediate
        device = volume.device

        grad_np = grad_sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_views = ang_np.shape[0]

        d_grad_sino = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_vol_grad = cuda.to_device(np.zeros((Nx, Ny, Nz), dtype=_DTYPE))

        (grid, tpb) = _grid_3d(n_views, det_u, det_v)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)
        cz = _DTYPE((Nz - 1) * 0.5)

        ConeProjectorFunction._backward_kernel[grid, tpb](
            d_grad_sino,
            n_views, det_u, det_v,
            Nx, Ny, Nz,
            d_vol_grad,
            _DTYPE(du), _DTYPE(dv),
            d_cos, d_sin,
            _DTYPE(step_size),
            _DTYPE(src_dist), _DTYPE(iso_dist),
            cx, cy, cz
        )

        grad_vol = torch.as_tensor(d_vol_grad, device=device)
        return grad_vol, None, None, None, None, None, None, None, None, None, None, None


# ############################################################################
# Cone‑beam: differentiable back‑projector
# ############################################################################
class ConeBackprojectorFunction(torch.autograd.Function):
    @_FASTMATH_DECORATOR
    def _forward_kernel(
        d_sino,
        n_views, n_u, n_v,
        Nx, Ny, Nz,
        d_reco,
        du, dv,
        d_cos, d_sin,
        step,
        src_dist, iso_dist,
        cx, cy, cz
    ):
        iview, iu, iv = cuda.grid(3)
        if iview >= n_views or iu >= n_u or iv >= n_v:
            return

        val   = d_sino[iview, iu, iv]
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]
        u     = (iu - (n_u - 1) * 0.5) * du
        v     = (iv - (n_v - 1) * 0.5) * dv

        sx = -iso_dist * sin_a
        sy =  iso_dist * cos_a
        sz = 0.0

        dx = (src_dist - iso_dist) * sin_a + u * cos_a
        dy = -(src_dist - iso_dist) * cos_a + u * sin_a
        dz = v

        rx = dx - sx
        ry = dy - sy
        rz = dz - sz
        length  = math.sqrt(rx * rx + ry * ry + rz * rz)
        inv_len = 1.0 / length
        rx *= inv_len
        ry *= inv_len
        rz *= inv_len

        t = 0.0
        while t < length:
            x  = sx + t * rx
            y  = sy + t * ry
            z  = sz + t * rz
            ix = x + cx
            iy = y + cy
            iz = z + cz
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            iz0 = int(math.floor(iz))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1 and 0 <= iz0 < Nz - 1:
                dx2 = ix - ix0
                dy2 = iy - iy0
                dz2 = iz - iz0
                cval = val * step
                cuda.atomic.add(d_reco, (ix0,     iy0,     iz0    ), cval * (1 - dx2) * (1 - dy2) * (1 - dz2))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0    ), cval * dx2       * (1 - dy2) * (1 - dz2))
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0    ), cval * (1 - dx2) * dy2       * (1 - dz2))
                cuda.atomic.add(d_reco, (ix0,     iy0,     iz0 + 1), cval * (1 - dx2) * (1 - dy2) * dz2)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0    ), cval * dx2       * dy2       * (1 - dz2))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0 + 1), cval * dx2       * (1 - dy2) * dz2)
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0 + 1), cval * (1 - dx2) * dy2       * dz2)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx2       * dy2       * dz2)
            t += step

    @_FASTMATH_DECORATOR
    def _backward_kernel(
        d_grad_out,
        n_views, n_u, n_v,
        Nx, Ny, Nz,
        d_sino_grad,
        du, dv,
        d_cos, d_sin,
        step,
        src_dist, iso_dist,
        cx, cy, cz
    ):
        iview, iu, iv = cuda.grid(3)
        if iview >= n_views or iu >= n_u or iv >= n_v:
            return

        cos_a = d_cos[iview]
        sin_a = d_sin[iview]
        u     = (iu - (n_u - 1) * 0.5) * du
        v     = (iv - (n_v - 1) * 0.5) * dv

        sx = -iso_dist * sin_a
        sy =  iso_dist * cos_a
        sz = 0.0

        dx = (src_dist - iso_dist) * sin_a + u * cos_a
        dy = -(src_dist - iso_dist) * cos_a + u * sin_a
        dz = v

        rx = dx - sx
        ry = dy - sy
        rz = dz - sz
        length  = math.sqrt(rx * rx + ry * ry + rz * rz)
        inv_len = 1.0 / length
        rx *= inv_len
        ry *= inv_len
        rz *= inv_len

        t = 0.0
        while t < length:
            x  = sx + t * rx
            y  = sy + t * ry
            z  = sz + t * rz
            ix = x + cx
            iy = y + cy
            iz = z + cz
            ix0 = int(math.floor(ix))
            iy0 = int(math.floor(iy))
            iz0 = int(math.floor(iz))
            if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1 and 0 <= iz0 < Nz - 1:
                dx2 = ix - ix0
                dy2 = iy - iy0
                dz2 = iz - iz0
                cval = (
                    d_grad_out[ix0,     iy0,     iz0    ] * (1 - dx2) * (1 - dy2) * (1 - dz2) +
                    d_grad_out[ix0 + 1, iy0,     iz0    ] * dx2       * (1 - dy2) * (1 - dz2) +
                    d_grad_out[ix0,     iy0 + 1, iz0    ] * (1 - dx2) * dy2       * (1 - dz2) +
                    d_grad_out[ix0,     iy0,     iz0 + 1] * (1 - dx2) * (1 - dy2) * dz2       +
                    d_grad_out[ix0 + 1, iy0 + 1, iz0    ] * dx2       * dy2       * (1 - dz2) +
                    d_grad_out[ix0 + 1, iy0,     iz0 + 1] * dx2       * (1 - dy2) * dz2       +
                    d_grad_out[ix0,     iy0 + 1, iz0 + 1] * (1 - dx2) * dy2       * dz2       +
                    d_grad_out[ix0 + 1, iy0 + 1, iz0 + 1] * dx2       * dy2       * dz2
                ) * step
                cuda.atomic.add(d_sino_grad, (iview, iu, iv), cval)
            t += step

    # ---------------- forward -----------------
    @staticmethod
    def forward(ctx, sinogram, angles,
                Nx, Ny, Nz,
                det_u, det_v, du, dv,
                step_size,
                source_distance, isocenter_distance):
        device = sinogram.device
        sino_np = sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_views = ang_np.shape[0]

        d_sino = cuda.to_device(sino_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_reco = cuda.to_device(np.zeros((Nx, Ny, Nz), dtype=_DTYPE))

        (grid, tpb) = _grid_3d(n_views, det_u, det_v)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)
        cz = _DTYPE((Nz - 1) * 0.5)

        ConeBackprojectorFunction._forward_kernel[grid, tpb](
            d_sino,
            n_views, det_u, det_v,
            Nx, Ny, Nz,
            d_reco,
            _DTYPE(du), _DTYPE(dv),
            d_cos, d_sin,
            _DTYPE(step_size),
            _DTYPE(source_distance), _DTYPE(isocenter_distance),
            cx, cy, cz
        )

        vol = torch.as_tensor(d_reco, device=device)
        ctx.save_for_backward(sinogram, angles)
        ctx.intermediate = (Nx, Ny, Nz, det_u, det_v, du, dv,
                            step_size, source_distance, isocenter_distance)
        return vol

    # -------------- backward ------------------
    @staticmethod
    def backward(ctx, grad_output):
        sinogram, angles = ctx.saved_tensors
        (Nx, Ny, Nz, det_u, det_v, du, dv,
         step_size, src_dist, iso_dist) = ctx.intermediate
        device = sinogram.device

        grad_np = grad_output.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np  = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_views = ang_np.shape[0]

        d_grad_out = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_sino_grad = cuda.to_device(np.zeros((n_views, det_u, det_v), dtype=_DTYPE))

        (grid, tpb) = _grid_3d(n_views, det_u, det_v)
        cx = _DTYPE((Nx - 1) * 0.5)
        cy = _DTYPE((Ny - 1) * 0.5)
        cz = _DTYPE((Nz - 1) * 0.5)

        ConeBackprojectorFunction._backward_kernel[grid, tpb](
            d_grad_out,
            n_views, det_u, det_v,
            Nx, Ny, Nz,
            d_sino_grad,
            _DTYPE(du), _DTYPE(dv),
            d_cos, d_sin,
            _DTYPE(step_size),
            _DTYPE(src_dist), _DTYPE(iso_dist),
            cx, cy, cz
        )
        grad_sino = torch.as_tensor(d_sino_grad, device=device)
        return grad_sino, None, None, None, None, None, None, None, None, None, None, None
