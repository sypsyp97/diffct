"""Measured CT-PD ingestion and focal-spot geometry on the portable backend."""

import builtins
import struct
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

import diffct


@pytest.fixture
def files(tmp_path):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, RawDataStorage, generate_uid

    result = []
    for index, instance in enumerate([7, 2, 3]):
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
            (0x7033, 0x100B): .02 if instance % 2 else -.01,
            (0x7033, 0x100C): 0. if instance % 2 else -.6,
            (0x7033, 0x100D): 0. if instance % 2 else .5,
        }
        for tag, value in values.items():
            d.add_new(tag, 'UN', struct.pack('<f', value))
        d.add_new((0x7031, 0x1033), 'UN', struct.pack('<2f', 3.7, 2.5))
        d.add_new((0x7029, 0x100B), 'UN', b'CYLINDRICAL ')
        d.add_new((0x7033, 0x100E), 'UN', b'FFSZ')
        d.add_new((0x7041, 0x1001), 'UN', b'0.0192')
        p = tmp_path / f'{index}.dcm'
        d.save_as(p, enforce_file_format=True)
        result.append(p)
    return result


def test_read_shuffled_sparse_views(files):
    scan = diffct.read_dicom_ct_pd(files[0].parent)
    assert isinstance(scan, diffct.ClinicalScan)
    assert scan.instance.tolist() == [2, 3, 7]
    assert scan.projections.dtype == np.float32
    assert scan.projections.shape == (3, 6, 4)
    np.testing.assert_allclose(scan.projections[0], (np.arange(24).reshape(6, 4) + 2) * .02 - .1, atol=1e-7)
    for name in ['angle', 'z', 'radius', 'sdd', 'focal_angle_offset', 'focal_z_offset', 'focal_radius_offset']:
        assert getattr(scan, name).dtype == np.float64
    np.testing.assert_allclose(scan.central_element, (2.7, 1.5))
    np.testing.assert_allclose(scan.detector_spacing, (1.2, 1.))
    assert scan.detector_shape == 'CYLINDRICAL'
    assert scan.ffs_mode == 'FFSZ'
    assert scan.water_attenuation == .0192
    assert scan.manufacturer == 'SIEMENS'
    assert scan.manufacturer_model_name == ''
    with pytest.raises(FrozenInstanceError):
        scan.ffs_mode = 'FFSNONE'


def test_file_list_and_duplicate(files):
    assert diffct.read_dicom_ct_pd(files[::-1]).instance.tolist() == [2, 3, 7]
    with pytest.raises(ValueError, match='duplicate'):
        diffct.read_dicom_ct_pd(files + files[:1])


def test_manufacturer_model_name_read_and_enforced_constant(files):
    import pydicom

    assert diffct.read_dicom_ct_pd(files).manufacturer_model_name == ''
    d = pydicom.dcmread(files[0])
    d.ManufacturerModelName = 'SOMATOM Example'
    d.save_as(files[0])
    with pytest.raises(ValueError, match='constant'):
        diffct.read_dicom_ct_pd(files)

    for path in files[1:]:
        d = pydicom.dcmread(path)
        d.ManufacturerModelName = 'SOMATOM Example'
        d.save_as(path)
    scan = diffct.read_dicom_ct_pd(files)
    assert scan.manufacturer_model_name == 'SOMATOM Example'
    assert diffct.probe_dicom_ct_pd(files).manufacturer_model_name == 'SOMATOM Example'

    d = pydicom.dcmread(files[0])
    d.ManufacturerModelName = 'Other model'
    d.save_as(files[0])
    with pytest.raises(ValueError, match='constant'):
        diffct.read_dicom_ct_pd(files)


def test_clinical_scan_model_defaults_to_empty(files):
    scan = diffct.read_dicom_ct_pd(files)
    legacy_names = ('projections', 'angle', 'z', 'radius', 'sdd', 'focal_angle_offset',
                    'focal_z_offset', 'focal_radius_offset', 'instance', 'detector_spacing',
                    'central_element', 'detector_shape', 'ffs_mode', 'water_attenuation',
                    'manufacturer')
    clone = diffct.ClinicalScan(**{name: getattr(scan, name) for name in legacy_names})
    assert clone.manufacturer_model_name == ''


def test_probe_uses_headers_only(files, monkeypatch):
    import pydicom

    calls = []
    original = pydicom.dcmread

    def tracking_dcmread(filename, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(pydicom, 'dcmread', tracking_dcmread)
    probe = diffct.probe_dicom_ct_pd(files[0].parent)

    assert isinstance(probe, diffct.ClinicalScanProbe)
    assert len(calls) == 3
    assert all(call.get('stop_before_pixels') is True for call in calls)
    assert probe.file_count == 3
    assert probe.instance_range == (2, 7)
    assert probe.detector_dimensions == (6, 4)
    assert probe.detector_spacing == pytest.approx((1.2, 1.))
    assert probe.sdd == pytest.approx(20.)
    assert probe.central_element == pytest.approx((2.7, 1.5))
    assert probe.detector_shape == 'CYLINDRICAL'
    assert probe.ffs_mode == 'FFSZ'
    assert probe.water_attenuation == pytest.approx(.0192)
    assert probe.manufacturer == 'SIEMENS'
    assert probe.manufacturer_model_name == ''


def test_ima_files_are_accepted(files):
    for path, suffix in zip(files, ('.ima', '.IMA', '.Ima')):
        path.rename(path.with_suffix(suffix))
    scan = diffct.read_dicom_ct_pd(files[0].parent)
    assert scan.instance.tolist() == [2, 3, 7]
    assert scan.projections.shape == (3, 6, 4)


def test_scan_angles_are_unwrapped(files):
    import pydicom

    raw = {2: 3.0, 3: -3.1, 7: -2.9}
    for path in files:
        d = pydicom.dcmread(path)
        angle = raw[int(d.InstanceNumber)]
        d[(0x7031, 0x1001)].value = struct.pack('<f', angle)
        d.save_as(path)

    scan = diffct.read_dicom_ct_pd(files)
    raw_sorted = np.array([raw[2], raw[3], raw[7]], dtype=np.float32).astype(np.float64)
    np.testing.assert_allclose(scan.angle, np.unwrap(raw_sorted), rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(scan.angle) > 0)
    np.testing.assert_allclose(np.sin(scan.angle), np.sin(raw_sorted), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.cos(scan.angle), np.cos(raw_sorted), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('tag,value', [
    ((0x7031, 0x1031), struct.pack('<f', 21.)),
    ((0x7029, 0x1002), struct.pack('<f', 1.5)),
    ((0x7031, 0x1033), struct.pack('<2f', 4., 2.5)),
    ((0x7029, 0x100B), b'FLAT'),
])
def test_nonconstant_geometry_rejected(files, tag, value):
    import pydicom
    d = pydicom.dcmread(files[0])
    d[tag].value = value
    d.save_as(files[0])
    with pytest.raises(ValueError, match='constant'):
        diffct.read_dicom_ct_pd(files)


def test_empty_input(tmp_path):
    with pytest.raises(ValueError, match='empty|No .*files'):
        diffct.read_dicom_ct_pd(tmp_path)


def test_missing_pydicom(monkeypatch, tmp_path):
    original = builtins.__import__
    def without_pydicom(name, *args, **kwargs):
        if name == 'pydicom':
            raise ImportError('not installed')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_pydicom)
    with pytest.raises(ImportError, match=r'diffct\[dicom\]'):
        diffct.read_dicom_ct_pd(tmp_path)
    with pytest.raises(ImportError, match=r'diffct\[dicom\]'):
        diffct.probe_dicom_ct_pd(tmp_path)


@pytest.mark.parametrize('ffs', [True, False])
def test_geometry_matches_hand_formulas(files, ffs):
    scan = diffct.read_dicom_ct_pd(files)
    g = diffct.clinical_geometry(scan, flying_focal_spot=ffs, dtype=torch.float64)
    assert isinstance(g, diffct.ClinicalGeometry)
    assert g.detector_shape == (6, 4)
    def t(name):
        return torch.tensor(getattr(scan, name))
    p, z, r = t('angle'), t('z'), t('radius')
    z = z - (z.min() + z.max()) / 2
    def cyl(r, p, z):
        return torch.stack((-r*p.sin(), r*p.cos(), z), 1)
    expected_source = cyl(r+t('focal_radius_offset')*ffs, p+t('focal_angle_offset')*ffs,
                          z+t('focal_z_offset')*ffs)
    zero = torch.zeros_like(p)
    expected = (expected_source, cyl(r, p, z)+t('sdd')[:, None]*torch.stack((p.sin(), -p.cos(), zero), 1),
                torch.stack((p.cos(), p.sin(), zero), 1), torch.stack((zero, zero, torch.ones_like(p)), 1))
    for actual, target in zip(g.trajectory, expected):
        torch.testing.assert_close(actual, target)
    u, v = torch.meshgrid(torch.tensor([-1., 0., 1.], dtype=torch.float64),
                          torch.tensor([-1., 1.], dtype=torch.float64), indexing='ij')
    off_u = (scan.central_element[0] - 2.5)*scan.detector_spacing[0]
    gamma = (u-off_u)/20
    expected_surface = torch.stack((20*gamma.sin(), -v, 20*(gamma.cos()-1)), -1)
    torch.testing.assert_close(g.detector_surface(u, v), expected_surface)


def test_center_overrides_and_flat_rejection(files):
    scan = diffct.read_dicom_ct_pd(files)
    g = diffct.clinical_geometry(scan, z_center=0., central_column=2.5, dtype=torch.float64)
    torch.testing.assert_close(g.trajectory[1][:, 2], torch.tensor(scan.z))
    uv = torch.zeros(1, dtype=torch.float64)
    torch.testing.assert_close(g.detector_surface(uv, uv), torch.zeros(1, 3, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match='FLAT|cylindrical'):
        diffct.clinical_geometry(replace(scan, detector_shape='FLAT'))


def test_portable_forward_and_adjoint(files):
    scan = diffct.read_dicom_ct_pd(files)
    operators = []
    for ffs in [True, False]:
        g = diffct.clinical_geometry(scan, flying_focal_spot=ffs, dtype=torch.float64)
        operators.append(diffct.Projector(g.trajectory, (3, 4, 4), g.detector_shape,
                         detector_spacing=g.detector_spacing, detector_surface=g.detector_surface,
                         backend='torch', precision='float64'))
    x = torch.arange(48, dtype=torch.float64).reshape(3, 4, 4)/48
    on, off = [op.project(x) for op in operators]
    assert torch.isfinite(on).all()
    assert not torch.allclose(on, off)
    y = torch.linspace(-1, 1, on.numel(), dtype=torch.float64).reshape(on.shape)
    torch.testing.assert_close((on*y).sum(), (x*operators[0].backproject(y)).sum(), rtol=1e-10, atol=1e-10)
