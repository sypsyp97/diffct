"""Read one DICOM-CT-PD projection series (and its image series) into npy/npz.

    python prep_case.py <projection dir> <image dir> <out dir>

out/proj.npy  (views, channels, rows) float32 line integrals, views sorted by InstanceNumber
out/meta.npz  per-view geometry: phi z rho sdd ce_col ce_row du dv dphi dz drho, plus ffs mode
out/ref.npz   vendor images in HU with z positions
"""
import glob, struct, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pydicom

proj_dir, img_dir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
f32 = lambda b: struct.unpack('<f', b[:4])[0]
NAMES = ['phi', 'z', 'rho', 'sdd', 'ce_col', 'ce_row', 'du', 'dv', 'dphi', 'dz', 'drho']


def header(d):
    g = lambda grp, el: d[grp, el].value
    ce = struct.unpack('<2f', g(0x7031, 0x1033)[:8])
    return (f32(g(0x7031, 0x1001)), f32(g(0x7031, 0x1002)), f32(g(0x7031, 0x1003)), f32(g(0x7031, 0x1031)),
            ce[0], ce[1], f32(g(0x7029, 0x1002)), f32(g(0x7029, 0x1006)),
            f32(g(0x7033, 0x100B)), f32(g(0x7033, 0x100C)), f32(g(0x7033, 0x100D)))


files = sorted(glob.glob(f'{proj_dir}/*.dcm'))
first = pydicom.dcmread(files[0])
ffs = first[0x7033, 0x100E].value
ffs = (ffs.decode() if isinstance(ffs, bytes) else str(ffs)).strip()
n_cols, n_rows = int(first.Rows), int(first.Columns)
print(first.Manufacturer, 'views', len(files), 'channels', n_cols, 'rows', n_rows, 'FFS', ffs, flush=True)


def read(p):
    d = pydicom.dcmread(p)
    proj = d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept)
    return int(d.InstanceNumber), proj, header(d)


with ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(read, files, chunksize=64))
order = np.argsort([r[0] for r in rows])
inst = np.array([rows[i][0] for i in order])
assert np.all(np.diff(inst) == 1), 'missing or duplicate InstanceNumber'
stack = np.lib.format.open_memmap(out / 'proj.npy', mode='w+', dtype=np.float32, shape=(len(rows), n_cols, n_rows))
for k, i in enumerate(order):
    stack[k] = rows[i][1]
stack.flush()
meta = np.array([rows[i][2] for i in order], dtype=np.float64)
for n, col in zip(NAMES, meta.T):
    print(f'{n:7s} min {col.min():.6g} max {col.max():.6g} unique {len(np.unique(col))}')
np.savez(out / 'meta.npz', instance=inst, ffs=ffs, manufacturer=str(first.Manufacturer),
         **{n: c for n, c in zip(NAMES, meta.T)})

ds = sorted((pydicom.dcmread(f) for f in glob.glob(f'{img_dir}/*.dcm')), key=lambda d: float(d.ImagePositionPatient[2]))
hu = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in ds])
np.savez(out / 'ref.npz', hu=hu, z=np.array([float(d.ImagePositionPatient[2]) for d in ds]),
         spacing=float(ds[0].PixelSpacing[0]), fov=float(ds[0].get('ReconstructionDiameter', 0) or 0),
         origin=np.array(ds[0].ImagePositionPatient, float))
print('ref images', hu.shape, 'z', ds[0].ImagePositionPatient[2], ds[-1].ImagePositionPatient[2], 'PREP_DONE', flush=True)
