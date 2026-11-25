"""Vendor-neutral DICOM-CT-PD projections and measured focal-spot geometry."""

import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


@dataclass(frozen=True)
class ClinicalScan:
    """Measured line integrals and CT-PD geometry, sorted by acquisition instance.

    ``projections`` is float32 ``(views, channels, rows)``. Geometry vectors
    are float64, distances are mm, angles are radians, and ``central_element``
    is a zero-based stored (column, row). Arrays remain caller-owned/mutable.
    ``water_attenuation`` converts reconstructed inverse-mm values to HU.
    """

    projections: np.ndarray
    angle: np.ndarray
    z: np.ndarray
    radius: np.ndarray
    sdd: np.ndarray
    focal_angle_offset: np.ndarray
    focal_z_offset: np.ndarray
    focal_radius_offset: np.ndarray
    instance: np.ndarray
    detector_spacing: tuple[float, float]
    central_element: tuple[float, float]
    detector_shape: str
    ffs_mode: str
    water_attenuation: float
    manufacturer: str


class ClinicalGeometry(NamedTuple):
    """Arguments for a cone ``Projector`` with a measured detector surface."""

    trajectory: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    detector_shape: tuple[int, int]
    detector_spacing: tuple[float, float]
    detector_surface: Callable


def _floats(dataset, tag, count=1):
    value = dataset[tag].value
    if isinstance(value, bytes):
        return struct.unpack('<' + 'f' * count, value[:4 * count])
    return (float(value),) if count == 1 else tuple(float(x) for x in value)


def _text(value):
    return (value.decode('ascii') if isinstance(value, bytes) else str(value)).strip(' \0')


def read_dicom_ct_pd(path):
    """Read a directory or iterable of CT-PD projection filenames.

    Install ``diffct[dicom]`` for this optional reader. Each file contains one
    view; InstanceNumber gaps are allowed but duplicates are rejected. Private
    geometry tags follow the TCIA DICOM-CT-PD format. Stored pixels are rescaled
    using each view's slope/intercept; no additional logarithm is applied.
    Detector dimensions, shape, spacing, distance and central element must be
    constant. Missing, invalid or inconsistent required metadata raises
    ``ValueError``. This reads vendor-neutral CT-PD, not proprietary raw files.
    """
    try:
        import pydicom
    except ImportError as error:
        raise ImportError('DICOM-CT-PD reading requires pip install "diffct[dicom]"') from error

    if isinstance(path, (str, Path)):
        directory = Path(path)
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() == '.dcm')
    else:
        files = list(path)
    if not files:
        raise ValueError('No DICOM-CT-PD files: empty input')

    names = ('angle', 'z', 'radius', 'sdd', 'focal_angle_offset', 'focal_z_offset', 'focal_radius_offset')
    tags = ((0x7031, 0x1001), (0x7031, 0x1002), (0x7031, 0x1003), (0x7031, 0x1031),
            (0x7033, 0x100B), (0x7033, 0x100C), (0x7033, 0x100D))
    rows, constants = [], None
    for filename in files:
        try:
            d = pydicom.dcmread(filename)
            geometry = tuple(_floats(d, tag)[0] for tag in tags)
            spacing = (_floats(d, (0x7029, 0x1002))[0], _floats(d, (0x7029, 0x1006))[0])
            central = tuple(x - 1 for x in _floats(d, (0x7031, 0x1033), 2))
            shape = _text(d[0x7029, 0x100B].value).upper()
            ffs = _text(d[0x7033, 0x100E].value)
            water = float(_text(d[0x7041, 0x1001].value))
            manufacturer = str(d.Manufacturer)
            dimensions = (int(d.Rows), int(d.Columns))
            values = (*geometry, *spacing, *central, water)
            if (not np.isfinite(values).all() or min(*spacing, water, geometry[2], geometry[3]) <= 0
                    or min(dimensions) <= 0):
                raise ValueError('geometry must be finite with positive dimensions, pitches and distances')
            current = (dimensions, shape, spacing, geometry[3], central, ffs, water, manufacturer)
            if constants is not None and current != constants:
                raise ValueError('detector geometry, FFS mode, water attenuation and manufacturer must be constant')
            constants = current
            pixels = d.pixel_array.astype(np.float32) * np.float32(d.RescaleSlope) + np.float32(d.RescaleIntercept)
            if pixels.shape != dimensions or not np.isfinite(pixels).all():
                raise ValueError('projection pixels must be finite and match detector dimensions')
            rows.append((int(d.InstanceNumber), pixels, geometry))
        except (KeyError, AttributeError, TypeError, ValueError, struct.error) as error:
            raise ValueError(f'Invalid DICOM-CT-PD file {filename}: {error}') from error
    rows.sort(key=lambda row: row[0])
    instance = np.array([row[0] for row in rows], dtype=np.int64)
    if np.any(np.diff(instance) == 0):
        raise ValueError('duplicate InstanceNumber in DICOM-CT-PD projections')
    vectors = np.array([row[2] for row in rows], dtype=np.float64)
    return ClinicalScan(np.stack([row[1] for row in rows]),
                        **dict(zip(names, vectors.T)), instance=instance,
                        detector_spacing=spacing, central_element=central, detector_shape=shape,
                        ffs_mode=ffs, water_attenuation=water, manufacturer=manufacturer)


def clinical_geometry(scan, *, flying_focal_spot=True, z_center=None, central_column=None,
                      device='cpu', dtype=torch.float32):
    """Build per-view cone geometry and a cylindrical stored-pixel surface.

    Detector frames use nominal focal centers; only the source includes FFS
    offsets. Stored rows run opposite to detector v. ``z_center`` defaults to
    the midpoint of nominal z. ``central_column`` overrides the zero-based
    stored central column, for calibrated geometry. Geometry is calculated in
    float64 then converted to ``device``/``dtype``. Flat/spherical detectors
    are not implemented. No patient-coordinate or HU conversion is performed.
    """
    if scan.detector_shape.upper() != 'CYLINDRICAL':
        raise NotImplementedError(f'clinical_geometry supports cylindrical detectors, not {scan.detector_shape}')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('clinical geometry dtype must be torch.float32 or torch.float64')
    if scan.projections.ndim != 3 or min(scan.projections.shape) == 0:
        raise ValueError('clinical projections must be nonempty (views, channels, rows)')
    count, channels, rows = scan.projections.shape
    names = ('angle', 'z', 'radius', 'sdd', 'focal_angle_offset', 'focal_z_offset', 'focal_radius_offset')
    vectors = [np.asarray(getattr(scan, name), dtype=np.float64) for name in names]
    if any(value.shape != (count,) or not np.isfinite(value).all() for value in vectors):
        raise ValueError('clinical geometry vectors must be finite with one value per view')
    if not np.all(vectors[3] == vectors[3][0]):
        raise ValueError('sdd must be constant')
    mid = (vectors[1].min() + vectors[1].max()) / 2 if z_center is None else float(z_center)
    column = scan.central_element[0] if central_column is None else float(central_column)
    du, dv = scan.detector_spacing
    if not np.isfinite((mid, column, *scan.central_element, du, dv)).all() or min(du, dv, vectors[3][0]) <= 0:
        raise ValueError('centers must be finite and detector pitches/distances positive')
    p, z, r, distance, dp, dz, dr = (torch.as_tensor(value, device=device, dtype=torch.float64) for value in vectors)
    z = z - mid
    def cyl(radius, angle, height):
        return torch.stack((-radius * angle.sin(), radius * angle.cos(), height), 1)
    source = cyl(r + dr, p + dp, z + dz) if flying_focal_spot else cyl(r, p, z)
    zero = torch.zeros_like(p)
    detector = cyl(r, p, z) + distance[:, None] * torch.stack((p.sin(), -p.cos(), zero), 1)
    trajectory = tuple(value.to(dtype=dtype) for value in (
        source, detector, torch.stack((p.cos(), p.sin(), zero), 1),
        torch.stack((zero, zero, torch.ones_like(p)), 1)))
    sdd = float(vectors[3][0])
    off_u = (column - (channels / 2 - .5)) * du
    off_v = (scan.central_element[1] - (rows / 2 - .5)) * dv

    def surface(u, v):
        gamma = (u - off_u) / sdd
        return torch.stack((sdd * gamma.sin(), -(v - off_v), sdd * (gamma.cos() - 1)), -1)

    return ClinicalGeometry(trajectory, (channels, rows), (du, dv), surface)
