"""Mayo Clinic public CT dataset discovery, loading and HU calibration.

Helpers for the two public Mayo / TCIA low-dose CT datasets:

* the Mayo 2020 "LDCT-and-Projection-data" release (DICOM-CT-PD projection
  folders plus reconstructed reference images), and
* the 2016 Low Dose CT Grand Challenge (AAPM-Mayo), after extracting its
  archives.

Projection reading lives in :func:`read_dicom_ct_pd`; this module adds the
folder conventions on top: :func:`discover_mayo_2020_cases` and
:func:`discover_mayo_challenge_2016_cases` locate each patient's full- and
reduced-dose projection and image folders, :func:`load_mayo_case` returns the
``ClinicalScan`` plus the vendor reference image series, and
:func:`read_dicom_ct_images` converts a CT image series to Hounsfield units.

Install ``diffct[dicom]`` for the DICOM-dependent functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class Mayo2020Case:
    """Projection and image folders of one Mayo 2020 patient, if present."""

    patient_id: str
    patient_root: Path
    projection_full_dose: Path | None
    projection_low_dose: Path | None
    image_full_dose: Path | None
    image_low_dose: Path | None


@dataclass(frozen=True)
class MayoChallenge2016Case:
    """Projection and image folders of one 2016 Grand Challenge patient."""

    patient_id: str
    projection_full_dose: Path | None
    projection_quarter_dose: Path | None
    image_full_dose: Path | None
    image_quarter_dose: Path | None


class CTImageSeries(NamedTuple):
    """A sorted DICOM CT image series in Hounsfield units.

    ``volume`` is float32 ``(slices, rows, columns)`` after the per-file
    rescale slope/intercept. ``slice_positions`` holds the
    ImagePositionPatient z coordinates in millimetres, sorted ascending;
    ``slice_spacing`` is their uniform step. ``pixel_spacing`` is the
    ``(row, column)`` pitch and ``slice_thickness`` the nominal slice width.
    """

    volume: np.ndarray
    slice_positions: np.ndarray
    slice_spacing: float
    pixel_spacing: tuple[float, float]
    slice_thickness: float


def discover_mayo_2020_cases(data_root):
    """List every patient below a Mayo 2020 data root with a usable case.

    A case is kept when at least one projection folder exists. Folder names
    are classified loosely: ``projection``/``image`` plus ``full``/``low``/
    ``quarter`` and ``dose``, so both ``full dose`` and ``FullDose``-style
    exports are found.
    """
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f'Mayo 2020 data root does not exist: {root}')
    cases = []
    for patient_root in sorted(path for path in root.iterdir() if path.is_dir()):
        found = {key: None for key in ('projection_full_dose', 'projection_low_dose',
                                       'image_full_dose', 'image_low_dose')}
        for folder in sorted(path for path in patient_root.rglob('*') if path.is_dir()):
            name = folder.name.casefold()
            is_full = 'full' in name and 'dose' in name
            is_low = ('low' in name or 'quarter' in name) and 'dose' in name
            if 'projection' in name:
                key = 'projection_full_dose' if is_full else (
                    'projection_low_dose' if is_low else None)
            elif 'image' in name:
                key = 'image_full_dose' if is_full else (
                    'image_low_dose' if is_low else None)
            else:
                key = None
            if key is not None and found[key] is None:
                found[key] = folder
        if not found['projection_full_dose'] and not found['projection_low_dose']:
            continue
        cases.append(Mayo2020Case(patient_id=patient_root.name,
                                  patient_root=patient_root, **found))
    return cases


def discover_mayo_challenge_2016_cases(data_root, *, split='train'):
    """List 2016 Grand Challenge patients in a ``train`` or ``test`` split.

    Expects the extracted dataset layout: ``Training_Projection_Data`` and
    ``Training_Image_Data`` (``Testing_*`` for ``split='test'``) with
    ``DICOM-CT-PD_FD``/``DICOM-CT-PD_QD`` projection folders per patient and
    ``full_1mm``/``quarter_1mm`` image folders per reconstruction kernel.
    The first kernel directory found is used for the image paths.
    """
    if split not in {'train', 'test'}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(data_root).expanduser()
    projection_root = root / ('Training_Projection_Data' if split == 'train'
                              else 'Testing_Projection_Data')
    image_root = root / ('Training_Image_Data' if split == 'train'
                         else 'Testing_Image_Data')
    if not projection_root.is_dir():
        raise FileNotFoundError(
            f'Mayo Challenge projection root does not exist: {projection_root}')
    cases = []
    for patient_root in sorted(path for path in projection_root.iterdir() if path.is_dir()):
        patient = patient_root.name
        image_full = image_quarter = None
        if image_root.is_dir():
            for kernel in sorted(path for path in image_root.iterdir() if path.is_dir()):
                full = (kernel / 'full_1mm' / patient / 'full_1mm' if split == 'train'
                        else kernel / patient / 'full_1mm')
                quarter = (kernel / 'quarter_1mm' / patient / 'quarter_1mm' if split == 'train'
                           else kernel / patient / 'quarter_1mm')
                image_full = full if image_full is None and full.is_dir() else image_full
                image_quarter = (quarter if image_quarter is None and quarter.is_dir()
                                 else image_quarter)
        full_projection = patient_root / 'DICOM-CT-PD_FD'
        quarter_projection = patient_root / 'DICOM-CT-PD_QD'
        cases.append(MayoChallenge2016Case(
            patient_id=patient,
            projection_full_dose=full_projection if full_projection.is_dir() else None,
            projection_quarter_dose=quarter_projection if quarter_projection.is_dir() else None,
            image_full_dose=image_full,
            image_quarter_dose=image_quarter,
        ))
    return cases


def _require_pydicom():
    try:
        import pydicom
    except ImportError as error:
        raise ImportError('DICOM reading requires pip install "diffct[dicom]"') from error
    return pydicom


def _dicom_files(path):
    if isinstance(path, (str, Path)):
        directory = Path(path)
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.iterdir()
                      if p.suffix.lower() in {'.dcm', '.ima'})
    return list(path)


def read_dicom_ct_images(path):
    """Read a directory or iterable of DICOM CT images into a HU volume.

    Each file contributes one axial slice. Slices are sorted by the z
    coordinate of ImagePositionPatient and converted with the per-slice
    RescaleSlope/RescaleIntercept, which yields Hounsfield units for the CT
    images distributed with the Mayo datasets. Slice spacing must be uniform;
    dimensions, pixel spacing and slice thickness must be constant.
    """
    pydicom = _require_pydicom()
    files = _dicom_files(path)
    if not files:
        raise ValueError('No DICOM image files: empty input')

    slices, positions, constants = [], [], None
    for filename in files:
        try:
            d = pydicom.dcmread(filename)
            position = float(d[0x0020, 0x0032].value[2])  # ImagePositionPatient z
            spacing = tuple(float(x) for x in d[0x0028, 0x0030].value)  # PixelSpacing
            thickness = float(d[0x0018, 0x0050].value)  # SliceThickness
            pixels = d.pixel_array.astype(np.float32)
            pixels = pixels * np.float32(d.RescaleSlope) + np.float32(d.RescaleIntercept)
            if pixels.ndim != 2 or not np.isfinite(pixels).all():
                raise ValueError('image pixels must be a finite 2D array')
            current = (pixels.shape, spacing, thickness)
            if constants is not None and current != constants:
                raise ValueError('image dimensions, spacing and thickness must be constant')
            constants = current
            if not np.isfinite(position) or min(spacing) <= 0 or thickness <= 0:
                raise ValueError('image geometry must be finite and positive')
            slices.append(pixels)
            positions.append(position)
        except (KeyError, AttributeError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f'Invalid DICOM image file {filename}: {error}') from error

    order = np.argsort(positions)
    positions = np.asarray(positions, dtype=np.float64)[order]
    volume = np.stack([slices[i] for i in order])
    if len(positions) > 1:
        steps = np.diff(positions)
        if np.any(steps <= 0):
            raise ValueError('duplicate ImagePositionPatient z in the image series')
        if not np.allclose(steps, steps.mean(), rtol=1e-4, atol=1e-4):
            raise ValueError('image slice spacing is not uniform')
        spacing_z = float(steps.mean())
    else:
        spacing_z = float(constants[2])
    return CTImageSeries(volume, positions, spacing_z,
                         pixel_spacing=constants[1], slice_thickness=constants[2])


def _dose_folder(case, dose, kind):
    attribute = f'{kind}_{dose}_dose'
    if not hasattr(case, attribute):
        valid = [name.removeprefix(f'{kind}_').removesuffix('_dose')
                 for name in vars(case) if name.startswith(f'{kind}_')]
        raise ValueError(f"dose must be one of {sorted(valid)} for {type(case).__name__}, "
                         f'got {dose!r}')
    return getattr(case, attribute)


def load_mayo_case(case, dose='full', *, reference_images=True):
    """Load one discovered Mayo case as ``(ClinicalScan, CTImageSeries | None)``.

    ``dose`` selects the dose level recorded on the case: ``'full'`` plus
    ``'low'`` for Mayo 2020 cases or ``'quarter'`` for 2016 cases.
    ``reference_images=False`` skips reading the vendor reconstruction.
    """
    from .clinical import read_dicom_ct_pd

    projection = _dose_folder(case, dose, 'projection')
    if projection is None:
        raise ValueError(f'case {case.patient_id} has no {dose}-dose projection folder')
    scan = read_dicom_ct_pd(projection)
    images = None
    if reference_images:
        image = _dose_folder(case, dose, 'image')
        if image is not None:
            images = read_dicom_ct_images(image)
    return scan, images


def attenuation_to_hu(values, water_attenuation):
    """Convert absolute linear attenuation coefficients to Hounsfield units.

    Applies ``1000 * (mu - mu_w) / mu_w``, the calibration the TCIA datasets
    document through the WaterAttenuationCoefficient tag exposed as
    ``ClinicalScan.water_attenuation``. Use this for reconstructions in
    absolute inverse-mm attenuation. If measurements were explicitly divided
    by ``water_attenuation`` before reconstruction, the result is instead in
    water units and its HU values are ``1000 * (reconstruction - 1)``.
    Accepts NumPy arrays, PyTorch tensors and scalars; the input type is
    preserved.
    """
    water = float(water_attenuation)
    if not np.isfinite(water) or water == 0.0:
        raise ValueError('water_attenuation must be finite and non-zero')
    return 1000.0 * (values - water) / water


def hu_to_attenuation(values, water_attenuation):
    """Convert Hounsfield units to absolute linear attenuation coefficients.

    Applies ``mu = mu_w * (1 + HU / 1000)``, the inverse of
    :func:`attenuation_to_hu`. The result is in the same inverse-length unit
    as ``water_attenuation``. Accepts NumPy arrays, PyTorch tensors and
    scalars; the input type is preserved.
    """
    water = float(water_attenuation)
    if not np.isfinite(water) or water == 0.0:
        raise ValueError('water_attenuation must be finite and non-zero')
    return water + water * values / 1000.0


__all__ = [
    'CTImageSeries',
    'Mayo2020Case',
    'MayoChallenge2016Case',
    'attenuation_to_hu',
    'hu_to_attenuation',
    'discover_mayo_2020_cases',
    'discover_mayo_challenge_2016_cases',
    'load_mayo_case',
    'read_dicom_ct_images',
]
