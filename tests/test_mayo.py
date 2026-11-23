"""Mayo dataset discovery, image-series reading and HU calibration."""

import builtins
import struct

import numpy as np
import pytest
import torch

import diffct


def _ct_pd_files(directory, instances):
    """Write minimal valid DICOM-CT-PD projection files, like test_clinical."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, RawDataStorage, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    for index, instance in enumerate(instances):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = RawDataStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        d = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
        d.SOPClassUID = meta.MediaStorageSOPClassUID
        d.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        d.InstanceNumber = instance
        d.Manufacturer = 'SIEMENS'
        d.Rows, d.Columns = 6, 4
        d.SamplesPerPixel, d.PhotometricInterpretation = 1, 'MONOCHROME2'
        d.BitsAllocated, d.BitsStored, d.HighBit, d.PixelRepresentation = 16, 16, 15, 0
        d.RescaleSlope, d.RescaleIntercept = .02, -.1
        d.PixelData = (np.arange(24, dtype=np.uint16) + instance).tobytes()
        values = {
            (0x7031, 0x1001): .3 * instance,
            (0x7031, 0x1002): .1 * instance,
            (0x7031, 0x1003): 10., (0x7031, 0x1031): 20.,
            (0x7029, 0x1002): 1.2, (0x7029, 0x1006): 1.,
            (0x7033, 0x100B): 0., (0x7033, 0x100C): -.6, (0x7033, 0x100D): .5,
        }
        for tag, value in values.items():
            d.add_new(tag, 'UN', struct.pack('<f', value))
        d.add_new((0x7031, 0x1033), 'UN', struct.pack('<2f', 3.7, 2.5))
        d.add_new((0x7029, 0x100B), 'UN', b'CYLINDRICAL')
        d.add_new((0x7033, 0x100E), 'UN', b'FFSZ')
        d.add_new((0x7041, 0x1001), 'UN', b'0.0192')
        d.save_as(directory / f'{index}.dcm', enforce_file_format=True)
    return directory


def _ct_image(directory, z, fill, *, slope=1., intercept=-1024.):
    """Write one minimal axial CT image DICOM; returns its path."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    d = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
    d.SOPClassUID = meta.MediaStorageSOPClassUID
    d.InstanceNumber = int(z * 10)
    d.Manufacturer = 'SIEMENS'
    d.Rows, d.Columns = 2, 3
    d.SamplesPerPixel, d.PhotometricInterpretation = 1, 'MONOCHROME2'
    d.BitsAllocated, d.BitsStored, d.HighBit, d.PixelRepresentation = 16, 16, 15, 0
    d.RescaleSlope, d.RescaleIntercept = slope, intercept
    d.PixelData = (np.full((2, 3), fill, dtype=np.uint16)).tobytes()
    d.add_new((0x0020, 0x0032), 'DS', [r'-160.0', '-160.0', f'{z:.6f}'])
    d.add_new((0x0028, 0x0030), 'DS', ['0.7', '0.7'])
    d.add_new((0x0018, 0x0050), 'DS', '1.0')
    path = directory / f'z{z}_{fill}.dcm'
    d.save_as(path, enforce_file_format=True)
    return path


def test_discover_mayo_2020(tmp_path):
    patient = tmp_path / 'L506'
    full_proj = patient / 'nested' / 'full dose projection'
    low_proj = patient / 'low dose projection'
    full_img = patient / 'full dose image'
    low_img = patient / 'low dose image'
    for folder in (full_proj, low_proj, full_img, low_img):
        folder.mkdir(parents=True)
    (patient / 'notes.txt').write_text('x')
    (tmp_path / 'images only' / 'full dose image').mkdir(parents=True)

    cases = diffct.discover_mayo_2020_cases(tmp_path)
    assert [c.patient_id for c in cases] == ['L506']
    case = cases[0]
    assert case.projection_full_dose == full_proj
    assert case.projection_low_dose == low_proj
    assert case.image_full_dose == full_img
    assert case.image_low_dose == low_img
    with pytest.raises(FileNotFoundError):
        diffct.discover_mayo_2020_cases(tmp_path / 'missing')


def test_discover_mayo_challenge_2016(tmp_path):
    proj_root = tmp_path / 'Training_Projection_Data' / 'P1'
    (proj_root / 'DICOM-CT-PD_FD').mkdir(parents=True)
    (proj_root / 'DICOM-CT-PD_QD').mkdir(parents=True)
    img_root = tmp_path / 'Training_Image_Data' / 'B30'
    (img_root / 'full_1mm' / 'P1' / 'full_1mm').mkdir(parents=True)
    (img_root / 'quarter_1mm' / 'P1' / 'quarter_1mm').mkdir(parents=True)

    (case,) = diffct.discover_mayo_challenge_2016_cases(tmp_path, split='train')
    assert case.patient_id == 'P1'
    assert case.projection_full_dose.name == 'DICOM-CT-PD_FD'
    assert case.projection_quarter_dose.name == 'DICOM-CT-PD_QD'
    assert case.image_full_dose.name == 'full_1mm'
    assert case.image_quarter_dose.name == 'quarter_1mm'
    with pytest.raises(ValueError, match='split'):
        diffct.discover_mayo_challenge_2016_cases(tmp_path, split='validation')
    with pytest.raises(FileNotFoundError):
        diffct.discover_mayo_challenge_2016_cases(tmp_path, split='test')


def test_read_dicom_ct_images_sorts_and_converts(tmp_path):
    for z, fill in [(10., 2100), (0., 1000), (5., 1024)]:
        _ct_image(tmp_path / 'img', z, fill)
    series = diffct.read_dicom_ct_images(tmp_path / 'img')
    assert series.volume.shape == (3, 2, 3)
    assert series.volume.dtype == np.float32
    np.testing.assert_allclose(series.slice_positions, [0., 5., 10.])
    np.testing.assert_allclose(series.volume[:, 0, 0], [-24., 0., 1076.])  # fill*1-1024
    assert series.slice_spacing == pytest.approx(5.)
    assert series.pixel_spacing == pytest.approx((.7, .7))
    assert series.slice_thickness == pytest.approx(1.)


def test_read_dicom_ct_images_rejects_bad_series(tmp_path):
    _ct_image(tmp_path / 'dup', 0., 1000)
    _ct_image(tmp_path / 'dup', 0., 1100)
    with pytest.raises(ValueError, match='duplicate'):
        diffct.read_dicom_ct_images(tmp_path / 'dup')
    for z in (0., 2., 10.):  # gaps 2 and 8 -> not uniform
        _ct_image(tmp_path / 'ragged', z, 1000)
    with pytest.raises(ValueError, match='uniform'):
        diffct.read_dicom_ct_images(tmp_path / 'ragged')
    with pytest.raises(ValueError, match='empty'):
        diffct.read_dicom_ct_images(tmp_path / 'empty')


def test_missing_pydicom(monkeypatch, tmp_path):
    original = builtins.__import__
    def without_pydicom(name, *args, **kwargs):
        if name == 'pydicom':
            raise ImportError('not installed')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_pydicom)
    with pytest.raises(ImportError, match=r'diffct\[dicom\]'):
        diffct.read_dicom_ct_images(tmp_path)


def test_load_mayo_case(tmp_path):
    patient = tmp_path / 'L506'
    _ct_pd_files(patient / 'full dose projection', [1, 2, 3])
    _ct_pd_files(patient / 'low dose projection', [4, 5])
    for z in (0., 1., 2.):
        _ct_image(patient / 'full dose image', z, 1000)
    (case,) = diffct.discover_mayo_2020_cases(tmp_path)

    scan, images = diffct.load_mayo_case(case)
    assert isinstance(scan, diffct.ClinicalScan)
    assert scan.projections.shape == (3, 6, 4)
    assert images.volume.shape == (3, 2, 3)

    scan_low, images_low = diffct.load_mayo_case(case, 'low')
    assert scan_low.projections.shape == (2, 6, 4)
    assert images_low is None  # no low-dose image folder on this case

    _, no_images = diffct.load_mayo_case(case, reference_images=False)
    assert no_images is None
    with pytest.raises(ValueError, match='quarter'):
        diffct.load_mayo_case(case, 'quarter')


def test_attenuation_to_hu():
    assert diffct.attenuation_to_hu(.0192, .0192) == pytest.approx(0.)
    assert diffct.attenuation_to_hu(0., .0192) == pytest.approx(-1000.)
    assert diffct.attenuation_to_hu(.0384, .0192) == pytest.approx(1000.)
    arr = diffct.attenuation_to_hu(np.array([0., .0192]), .0192)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, [-1000., 0.])
    t = diffct.attenuation_to_hu(torch.tensor([0., .0192]), .0192)
    assert isinstance(t, torch.Tensor)
    torch.testing.assert_close(t, torch.tensor([-1000., 0.]))
    with pytest.raises(ValueError, match='water_attenuation'):
        diffct.attenuation_to_hu(1., 0.)
