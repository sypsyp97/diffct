"""Read CT-PD projections and vendor images into the gallery's npy/npz files.

    python docs/assets/gallery/ldct_prep.py <projection dir> <image dir> <out dir>
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pydicom

from diffct import read_dicom_ct_pd


def main():
    proj_dir, img_dir, output = sys.argv[1:]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    scan = read_dicom_ct_pd(proj_dir)
    n_cols, n_rows = scan.projections.shape[1:]
    print(scan.manufacturer, 'views', len(scan.instance), 'channels', n_cols, 'rows', n_rows,
          'FFS', scan.ffs_mode, flush=True)
    np.save(out / 'proj.npy', scan.projections)
    meta = {'phi': scan.angle, 'z': scan.z, 'rho': scan.radius, 'sdd': scan.sdd,
            'dphi': scan.focal_angle_offset, 'dz': scan.focal_z_offset, 'drho': scan.focal_radius_offset}
    for name, value in zip(('ce_col', 'ce_row', 'du', 'dv'),
                           (*[x + 1 for x in scan.central_element], *scan.detector_spacing)):
        meta[name] = np.full(len(scan.instance), value, dtype=np.float64)
    for name, values in meta.items():
        print(f'{name:7s} min {values.min():.6g} max {values.max():.6g} unique {len(np.unique(values))}')
    np.savez(out / 'meta.npz', **meta, instance=scan.instance, ffs=scan.ffs_mode,
             manufacturer=scan.manufacturer, detector_shape=scan.detector_shape,
             water_attenuation=scan.water_attenuation)

    ds = sorted((pydicom.dcmread(f) for f in glob.glob(f'{img_dir}/*.dcm')),
                key=lambda d: float(d.ImagePositionPatient[2]))
    hu = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope)
                   + float(d.RescaleIntercept) for d in ds])
    np.savez(out / 'ref.npz', hu=hu, z=np.array([float(d.ImagePositionPatient[2]) for d in ds]),
             spacing=float(ds[0].PixelSpacing[0]), fov=float(ds[0].get('ReconstructionDiameter', 0) or 0),
             origin=np.array(ds[0].ImagePositionPatient, float))
    print('ref images', hu.shape, 'z', ds[0].ImagePositionPatient[2], ds[-1].ImagePositionPatient[2],
          'PREP_DONE', flush=True)


if __name__ == '__main__':
    main()
