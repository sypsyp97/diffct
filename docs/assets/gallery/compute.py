"""Compute the arrays behind the documentation figures; render.py draws them.

Every array comes from a diffct run. The commands write compressed ``.npz``
files of display slices (float16) to ``--out``:

    python docs/assets/gallery/compute.py phantoms2d --ldct DIR --out OUT
    python docs/assets/gallery/compute.py adam --ldct DIR --lr 0.1 --out OUT
    python docs/assets/gallery/compute.py volumes --ldct DIR --out OUT
    python docs/assets/gallery/compute.py gradient --walnut examples/data/walnut_cone.npz --out OUT
    torchrun --nproc-per-node=8 docs/assets/gallery/compute.py helical \\
        --case DIR/L213 --center 442.2 --out OUT

``--ldct`` holds one directory per case written by ldct_prep.py. The patient
data are from LDCT-and-Projection-data (McCollough et al., The Cancer Imaging
Archive, https://doi.org/10.7937/9npb-2637, CC BY 4.0); see
docs/assets/gallery/NOTICE. ``helical`` reconstructs measured helical
projections; the other commands use the vendor images or Shepp-Logan as
ground truth for simulated scans.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

from diffct import (  # noqa: E402
    Projector,
    cgls,
    circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    circular_trajectory_3d,
    fbp,
    fdk,
)
from _common import Scan, psnr, shepp_logan_3d  # noqa: E402

MU_WATER = 0.0186  # 1/mm; volumes are in water units, HU = 1000 (x - 1)
WINDOW = "shepp-logan"


def half(x):
    return x.detach().float().cpu().numpy().astype(np.float16)


def water_units(hu):
    return np.clip(1.0 + hu / 1000.0, 0.0, None).astype(np.float32)


def ref_slice(case, z_mm=None):
    ref = np.load(case / "ref.npz")
    index = len(ref["z"]) // 2 if z_mm is None else int(np.argmin(np.abs(ref["z"] - z_mm)))
    return water_units(ref["hu"][index]), float(ref["z"][index])


def save(out, name, **arrays):
    np.savez_compressed(out / f"{name}.npz", **arrays)
    print("saved", name, {k: getattr(v, "shape", v) for k, v in arrays.items()}, flush=True)


def analytical_2d(truth, views, device):
    """Parallel FBP over 180 degrees and fan FBP over 360 degrees, geometry as in
    examples/analytical_reconstruction.py."""
    n = truth.shape[-1]
    result = {}
    trajectory = circular_trajectory_2d_parallel(views, end_angle=math.pi, device="cpu")
    sinogram = Projector(trajectory, (n, n), 3 * n, beam="parallel", detector_spacing=0.5).project(truth)
    angles = torch.linspace(0.0, math.pi, views + 1, device=device)[:-1]
    image = fbp(sinogram, trajectory, (n, n), angles, detector_spacing=0.5, window=WINDOW).clamp_min(0)
    result["parallel"] = (sinogram, image)
    sid, sdd, pitch = 2.5 * n, 4.0 * n, 0.8
    trajectory = circular_trajectory_2d_fan(views, sid, sdd, device="cpu")
    sinogram = Projector(trajectory, (n, n), 3 * n, beam="fan", detector_spacing=pitch).project(truth)
    angles = torch.linspace(0.0, 2 * math.pi, views + 1, device=device)[:-1]
    image = fbp(sinogram, trajectory, (n, n), angles, beam="fan", detector_spacing=pitch,
                window=WINDOW, isocenter=(0, 0)).clamp_min(0)
    result["fan"] = (sinogram, image)
    return result


def adam_loop(truth, beam, views, device, steps=200, keep=(20, 50, 200), lr=1e-2):
    """The least-squares loop of the iterative example pages."""
    n = truth.shape[-1]
    if beam == "parallel":
        trajectory = circular_trajectory_2d_parallel(views, device="cpu")
        operator = Projector(trajectory, (n, n), 3 * n, beam="parallel", detector_spacing=0.5)
    else:
        trajectory = circular_trajectory_2d_fan(views, 2.5 * n, 4.0 * n, device="cpu")
        operator = Projector(trajectory, (n, n), 3 * n, beam="fan", detector_spacing=0.8)
    measured = operator.project(truth).detach()
    image = torch.zeros((n, n), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=lr)
    losses, snapshots = [], {}
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss = 0.5 * (operator.project(image) - measured).square().sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            image.clamp_(min=0)
        losses.append(loss.item())
        if step in keep:
            snapshots[step] = image.detach().clone()
    return snapshots, np.array(losses)


def learn_filter(truth, views, device, steps=200, noise=0.02):
    """A learnable gain on the ramp filter, trained through fbp() on noisy data."""
    torch.manual_seed(7)
    n = truth.shape[-1]
    trajectory = circular_trajectory_2d_parallel(views, end_angle=math.pi, device="cpu")
    clean = Projector(trajectory, (n, n), 3 * n, beam="parallel", detector_spacing=0.5).project(truth)
    measurements = clean + torch.randn_like(clean) * (noise * clean.abs().max())
    angles = torch.arange(views, device=device) * (math.pi / views)
    fft_size = 2 * measurements.shape[1]
    lag = torch.arange(fft_size, device=device)
    lag = torch.minimum(lag, fft_size - lag)
    impulse = torch.zeros(fft_size, device=device)
    impulse[0] = math.pi / 2
    odd = lag % 2 == 1
    impulse[odd] = -2 / (math.pi * lag[odd].float().square())
    ramp = torch.fft.rfft(impulse).real / 0.5
    log_gain = torch.nn.Parameter(torch.zeros_like(ramp))

    def reconstruct():
        return fbp(measurements, trajectory, (n, n), angles, detector_spacing=0.5,
                   filter_response=ramp * log_gain.exp())

    with torch.no_grad():
        before = reconstruct()
    optimizer = torch.optim.Adam([log_gain], lr=0.02)
    for _ in range(steps):
        optimizer.zero_grad()
        (reconstruct() - truth).square().mean().backward()
        optimizer.step()
    with torch.no_grad():
        after = reconstruct()
    return before, after, ramp, ramp * log_gain.exp()


def phantoms2d(args):
    device = torch.device("cuda")
    out = Path(args.out)
    n = 512
    truths = {
        "shepp": (torch.from_numpy(shepp_logan_3d(n)[n // 2]), "Shepp-Logan"),
        "abdomen": (torch.from_numpy(ref_slice(Path(args.ldct) / "L213", -364.0)[0]), "TCIA L213, abdomen"),
        "chest": (torch.from_numpy(ref_slice(Path(args.ldct) / "C002")[0]), "TCIA C002, chest"),
    }
    pairs = {"parallel": ("shepp", "abdomen"), "fan": ("shepp", "chest")}
    for key, (truth, label) in truths.items():
        truth = truth.to(device)
        for beam, (sinogram, image) in analytical_2d(truth, 720, device).items():
            if key in pairs[beam]:
                save(out, f"fbp_{beam}_{key}", truth=half(truth), sinogram=half(sinogram[::2, ::2]),
                     image=half(image), psnr=psnr(image, truth), label=label, views=720)
    truth = truths["abdomen"][0].to(device)
    small = torch.nn.functional.avg_pool2d(truth[None, None], 2)[0, 0]
    before, after, ramp, learned = learn_filter(small, 360, device)
    save(out, "learning", truth=half(small), before=half(before), after=half(after), ramp=half(ramp),
         learned=half(learned), psnr=np.array([psnr(before, small), psnr(after, small)]), steps=200,
         noise=0.02, views=360)


def adam(args):
    """The Adam loop of the gradient-based page on the 512² abdominal slice, for one learning rate."""
    device = torch.device("cuda")
    truth = torch.from_numpy(ref_slice(Path(args.ldct) / "L213", -364.0)[0]).to(device)
    keep = tuple(sorted(set(args.keep) | {args.steps}))
    for beam in ("parallel", "fan"):
        start = time.time()
        snapshots, losses = adam_loop(truth, beam, 720, device, steps=args.steps, keep=keep, lr=args.lr)
        save(Path(args.out), f"iterative_{beam}_lr{args.lr:g}", truth=half(truth), losses=losses,
             steps=np.array(list(snapshots)), images=np.stack([half(v) for v in snapshots.values()]),
             psnr=np.array([psnr(v, truth) for v in snapshots.values()]), lr=args.lr, views=720,
             seconds=time.time() - start)


def volumes(args):
    device = torch.device("cuda")
    out = Path(args.out)
    if not args.curved_only:
        cone_phantom(device, out)
    curved_patient(device, out, Path(args.ldct))


def cone_phantom(device, out, views=720):
    n = 256
    phantom = torch.from_numpy(shepp_logan_3d(n)).to(device)
    scan = Scan(n, views)
    trajectory = scan.trajectory("circular")
    sinogram = Projector(trajectory, (n, n, n), scan.detector, detector_spacing=scan.pitch).project(phantom)
    angles = torch.linspace(0.0, 2 * math.pi, scan.views + 1, device=device)[:-1]
    volume = fdk(sinogram, trajectory, (n, n, n), angles, detector_spacing=scan.pitch, window=WINDOW,
                 isocenter=(0, 0, 0)).clamp_min(0)
    off = n // 2 + n // 8  # z = 0.25 of the half height, through the three small ellipsoids
    save(out, "fdk_cone", projection=half(sinogram[0].T), axial=half(volume[n // 2]),
         off_axial=half(volume[off]), coronal=half(volume[:, n // 2]),
         phantom_axial=half(phantom[n // 2]), phantom_off_axial=half(phantom[off]),
         phantom_coronal=half(phantom[:, n // 2]), psnr=psnr(volume, phantom), views=views, size=n,
         off_index=off)


def curved_patient(device, out, ldct):
    """Patient volume (vendor images) projected onto a cylindrical detector centred on the source."""
    ref = np.load(ldct / "L213" / "ref.npz")
    hu, z = ref["hu"], ref["z"]
    dz, dxy = float(abs(np.median(np.diff(z)))), float(ref["spacing"])
    body = torch.from_numpy(water_units(hu)).to(device)
    sid, sdd, du, dv = 541.0, 949.0, 1.0, 1.0
    depth = body.shape[0] * dz
    rows = int(math.ceil(depth * sdd / (sid - 0.5 * 512 * dxy) / dv)) + 8
    views = [0, 60, 120]  # degrees; only the shown views are projected
    trajectory = tuple(t[views] for t in circular_trajectory_3d(360, sid=sid, sdd=sdd, device="cpu"))

    def cylinder(u, v):
        angle = u / sdd
        return torch.stack((sdd * angle.sin(), v, sdd * (angle.cos() - 1)), dim=-1)

    operator = Projector(trajectory, tuple(body.shape), (888, rows), detector_spacing=(du, dv),
                         voxel_spacing=(dz, dxy, dxy), detector_surface=cylinder)
    sinogram = operator.project(body)
    save(out, "curved_detector", projections=np.stack([half(p.T) for p in sinogram]),
         angles=np.array(views) * 1.0, rows=rows, sid=sid, sdd=sdd)


def gradient(args):
    """Gradient-based reconstruction of the measured walnut: the volume is a parameter, the
    differentiable forward projector maps it to projections, and Adam minimises the MSE
    against the measurement."""
    from walnut_reconstruction import circular_geometry, load_scan, shell_scale
    device = torch.device("cuda")
    out = Path(args.out)
    sinogram, angles, sid, sdd, du, dv, offset_u, offset_v = load_scan(Path(args.walnut))
    n = 256
    voxel = 256 * du * sid / sdd / n
    measured = torch.from_numpy(sinogram).to(device)
    trajectory = circular_geometry(angles, sid, sdd, offset_u, offset_v)
    reference = fdk(measured, trajectory, (n,) * 3, angles, detector_spacing=(du, dv), voxel_spacing=voxel,
                    window="hann", isocenter=(0, 0, 0)).clamp_min(0)
    scale = shell_scale(reference)
    measured = measured * scale
    operator = Projector(trajectory, (n,) * 3, sinogram.shape[1:], detector_spacing=(du, dv), voxel_spacing=voxel)
    volume = torch.zeros((n,) * 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([volume], lr=args.lr)
    keep, snapshots, losses = set(args.keep), {}, []
    for step in range(1, max(args.keep) + 1):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(operator.project(volume), measured)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            volume.clamp_(min=0)
        losses.append(loss.item())
        if step in keep:
            snapshots[step] = volume.detach().clone()
    arrays = {}
    for step, v in snapshots.items():
        arrays[f"axial_{step}"] = half(v[n // 2])
        arrays[f"coronal_{step}"] = half(v[:, n // 2])
    save(out, "gradient_walnut", **arrays, steps=np.array(sorted(snapshots)), losses=np.array(losses),
         fdk_axial=half(reference[n // 2] * scale), fdk_coronal=half(reference[:, n // 2] * scale),
         lr=args.lr, views=len(angles), size=n)


def helical(args):
    """CGLS of measured full-dose helical projections with diffct's distributed Projector."""
    from torch import distributed as dist
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    case = Path(args.case)
    m = dict(np.load(case / "meta.npz"))
    proj = np.load(case / "proj.npy", mmap_mode="r")
    (n_cols, n_rows), z = proj.shape[1:], m["z"]
    sdd, du, dv, rho = (float(m[k][0]) for k in ("sdd", "du", "dv", "rho"))
    margin = math.ceil(n_rows / 2 * dv * (rho + 250) / sdd + np.abs(m["dz"]).max())
    mid = 0.5 * (z.min() + z.max())
    depth = int(math.ceil(z.max() - z.min() + 2 * margin))
    off_u = args.center - (n_cols / 2 - 0.5)
    off_v = (float(m["ce_row"][0]) - 1) - (n_rows / 2 - 0.5)

    def cyl(r, phi, height):
        return torch.stack((-r * phi.sin(), r * phi.cos(), height), 1)

    t = lambda k: torch.as_tensor(m[k], dtype=torch.float64)  # noqa: E731
    phi, zz, r = t("phi"), t("z") - mid, t("rho")
    source = cyl(r + t("drho"), phi + t("dphi"), zz + t("dz"))  # flying focal spot
    s, c, zero = phi.sin(), phi.cos(), torch.zeros_like(phi)
    detector = cyl(r, phi, zz) + sdd * torch.stack((s, -c, zero), 1)
    trajectory = tuple(x.float() for x in (source, detector, torch.stack((c, s, zero), 1),
                                            torch.stack((zero, zero, torch.ones_like(phi)), 1)))

    def surface(u, v):  # cylinder centred on the nominal focal spot; stored rows run against v
        gamma = (u - off_u * du) / sdd
        return torch.stack((sdd * gamma.sin(), -(v - off_v * dv), sdd * (gamma.cos() - 1)), -1)

    operator = Projector(trajectory, (depth, 512, 512), (n_cols, n_rows), detector_spacing=(du, dv),
                         voxel_spacing=1.0, detector_surface=surface, distributed=True, partition="views")
    b = torch.from_numpy(np.ascontiguousarray(proj[operator.view_slice])).to(device).div_(MU_WATER)
    keep, snapshots = set(args.keep), {}

    def callback(iteration, residual, volume):
        if rank == 0 and iteration in keep:
            snapshots[iteration] = volume.detach().float().cpu()
            print(f"it {iteration} residual {residual:.6g}", flush=True)

    start = time.time()
    result = cgls(operator, b, max(args.keep), callback=callback)
    torch.cuda.synchronize()
    seconds = time.time() - start
    if rank == 0:
        axial_z = args.axial_z or [mid - 60.0, mid, mid + 60.0]
        axial_index = [int(round(zv - (mid - depth / 2))) for zv in axial_z]
        arrays = {}
        for it, volume in snapshots.items():
            arrays[f"axial_{it}"] = np.stack([volume[k].numpy().astype(np.float16) for k in axial_index])
            arrays[f"coronal_{it}"] = volume[:, 256].numpy().astype(np.float16)
            arrays[f"sagittal_{it}"] = volume[:, :, 256].numpy().astype(np.float16)
        history = np.array([float(x) for x in result.residual_history])
        save(Path(args.out), f"helical_{case.name}", **arrays, iterations=np.array(sorted(snapshots)),
             residual=history, axial_z=np.array(axial_z), seconds=seconds,
             views=len(z), detector=np.array([n_cols, n_rows]), gpus=dist.get_world_size(),
             manufacturer=str(m["manufacturer"]), depth=depth, z_mid=mid)
        print("HELICAL_DONE", case.name, f"{seconds:.1f} s", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("phantoms2d", "volumes"):
        p = sub.add_parser(name)
        p.add_argument("--ldct", required=True)
        p.add_argument("--out", required=True)
        if name == "volumes":
            p.add_argument("--curved-only", action="store_true", help="skip the Shepp-Logan FDK volume")
    p = sub.add_parser("adam")
    p.add_argument("--ldct", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--keep", type=int, nargs="+", default=[50, 200])
    p.add_argument("--out", required=True)
    p = sub.add_parser("gradient")
    p.add_argument("--walnut", required=True, help="examples/data/walnut_cone.npz")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--keep", type=int, nargs="+", default=[10, 50, 200, 500])
    p.add_argument("--out", required=True)
    p = sub.add_parser("helical")
    p.add_argument("--case", required=True)
    p.add_argument("--center", type=float, required=True, help="0-based stored column of the central ray")
    p.add_argument("--keep", type=int, nargs="+", default=[15, 30, 60], help="CGLS iterations to save")
    p.add_argument("--axial-z", type=float, nargs="+", help="scanner z (mm) of the saved axial slices")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"phantoms2d": phantoms2d, "adam": adam, "volumes": volumes, "gradient": gradient, "helical": helical}[args.command](args)


if __name__ == "__main__":
    main()
