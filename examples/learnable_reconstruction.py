"""Build FBP/FDK in PyTorch and learn its filter and detector weights.

Synthetic training data; the fitted parameters belong to this phantom/noise.
    python examples/learnable_reconstruction.py --beam parallel --size 32
    python examples/learnable_reconstruction.py --beam cone --steps 20
"""

import argparse
import math

import torch
import torch.nn.functional as F

from diffct import (
    Projector, angular_integration_weights, circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel, circular_trajectory_3d,
    cone_cosine_weights, cone_weighted_backproject, fan_cosine_weights,
    fan_weighted_backproject, fbp, fdk, parallel_weighted_backproject,
)
from _common import psnr, shepp_logan_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beam", choices=("parallel", "fan", "cone"), default="parallel")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--views", type=int, default=180)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.size < 8 or args.views < 8 or args.steps < 1:
        parser.error("size/views must be >=8 and steps must be >=1")
    if not torch.cuda.is_available():
        raise RuntimeError("diffct needs a CUDA GPU")
    torch.manual_seed(7)
    n, beam, views = args.size, args.beam, args.views
    sid, sdd = 2.5 * n, 4.0 * n
    pitch = 0.5 if beam == "parallel" else 0.8
    detector = (3 * n, 2 * n) if beam == "cone" else 3 * n
    shape = (n,) * (3 if beam == "cone" else 2)
    end_angle = math.pi if beam == "parallel" else 2 * math.pi
    angles = torch.arange(views, device="cuda") * (end_angle / views)
    if beam == "parallel":
        trajectory = circular_trajectory_2d_parallel(views, end_angle=math.pi, device="cpu")
        cosine = torch.ones(detector, device="cuda")
    elif beam == "fan":
        trajectory = circular_trajectory_2d_fan(views, sid, sdd, device="cpu")
        cosine = fan_cosine_weights(detector, pitch, sdd, device="cuda")
    else:
        trajectory = circular_trajectory_3d(views, sid, sdd, device="cpu")
        cosine = cone_cosine_weights(*detector, pitch, pitch, sdd, device="cuda")
    truth = torch.from_numpy(shepp_logan_3d(n)).cuda()
    if beam != "cone":
        truth = truth[n // 2]
    operator = Projector(trajectory, shape, detector, beam=beam, detector_spacing=pitch)
    clean = operator.project(truth)
    measurements = clean + torch.randn_like(clean) * (0.02 * clean.abs().max())

    # Finite discrete Ram-Lak impulse response, matching ramp_filter_1d.
    # Units: the sampled frequency response includes 1 / detector pitch.
    n_u = measurements.shape[1]
    fft_size = 2 * n_u
    lag = torch.arange(fft_size, device="cuda")
    lag = torch.minimum(lag, fft_size - lag)
    impulse = torch.zeros(fft_size, device="cuda")
    impulse[0] = math.pi / 2
    odd = lag % 2 == 1
    impulse[odd] = -2 / (math.pi * lag[odd].float().square())
    ramp = torch.fft.rfft(impulse).real / pitch
    log_filter_gain = torch.nn.Parameter(torch.zeros_like(ramp))
    log_detector_gain = torch.nn.Parameter(torch.zeros_like(cosine))
    quadrature = angular_integration_weights(angles, redundant_full_scan=beam != "parallel")
    quadrature = quadrature.reshape((views,) + (1,) * (measurements.ndim - 1))

    def manual_reconstruction():
        # 1. A learnable multiplicative correction to geometric cosine weights.
        weighted = measurements * cosine * log_detector_gain.exp()
        # 2. Learn a real frequency response, then filter along detector u.
        pad = (0, 0, n_u // 2, n_u - n_u // 2) if beam == "cone" else (n_u // 2, n_u - n_u // 2)
        padded = F.pad(weighted, pad)
        response = ramp * log_filter_gain.exp()
        response_shape = [1] * measurements.ndim
        response_shape[1] = response.numel()
        filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=1) * response.reshape(response_shape),
                                  n=fft_size, dim=1).narrow(1, n_u // 2, n_u)
        # 3. Angular integration, then analytical gather with its own transpose.
        filtered = filtered * quadrature
        if beam == "parallel":
            return parallel_weighted_backproject(filtered, *trajectory, pitch, *shape)
        if beam == "fan":
            return fan_weighted_backproject(filtered, *trajectory, pitch, *shape, isocenter=(0, 0))
        return cone_weighted_backproject(filtered, *trajectory, *shape, pitch, pitch, isocenter=(0, 0, 0))

    with torch.no_grad():
        before = manual_reconstruction()
        function = fdk if beam == "cone" else fbp
        options = {} if beam == "cone" else {"beam": beam}
        packaged = function(measurements, trajectory, shape, angles, detector_spacing=pitch,
                            projection_weights=cosine * log_detector_gain.exp(),
                            filter_response=ramp * log_filter_gain.exp(),
                            isocenter=(0,) * len(shape), **options)
        print(f"manual/API maximum difference: {(before - packaged).abs().max().item():.3g}")
        initial_loss = (before - truth).square().mean().item()
        initial_psnr = psnr(before, truth)

    optimizer = torch.optim.Adam([log_filter_gain, log_detector_gain], lr=0.02)
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = (manual_reconstruction() - truth).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = manual_reconstruction()
        final_loss = (after - truth).square().mean().item()
    print(f"{beam}, {args.steps} steps: MSE {initial_loss:.6g} -> {final_loss:.6g}, "
          f"PSNR {initial_psnr:.2f} -> {psnr(after, truth):.2f} dB")
    print(f"gradient norms: filter {log_filter_gain.grad.norm().item():.6g}, "
          f"weight {log_detector_gain.grad.norm().item():.6g}")


if __name__ == "__main__":
    main()
