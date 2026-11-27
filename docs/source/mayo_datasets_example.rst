Mayo Public Datasets
====================

The Mayo loaders discover and read the two public patient datasets of the Mayo
Clinic CT team: the Low Dose CT Image and Projection Data collection (Mayo 2020,
see :ref:`data-sources`) and the 2016 Low Dose CT Grand Challenge. Both store
projections in the DICOM-CT-PD format that :func:`~diffct.read_dicom_ct_pd`
reads; the loaders add folder discovery, vendor-image reading and Hounsfield
calibration on top. They need the ``dicom`` extra: ``pip install 'diffct[dicom]'``.

Discovery
---------

Point the discovery helpers at the root of an extracted download. Patient
folders are classified loosely by name, so both TCIA-style
``1.000000-Full dose projections-22835`` and ``FullDose``-style exports are
found. Cases without any projection folder are skipped.

.. code-block:: python

   from diffct import discover_mayo_2020_cases, discover_mayo_challenge_2016_cases

   cases = discover_mayo_2020_cases("LDCT-and-Projection-data")
   case = cases[0]
   case.patient_id            # e.g. 'C002'
   case.projection_full_dose  # pathlib.Path, or None when absent
   case.projection_low_dose
   case.image_full_dose

For the 2016 Grand Challenge, pass the extracted dataset root and the split;
the projection folders are ``DICOM-CT-PD_FD``/``DICOM-CT-PD_QD`` per patient and
the image folders sit per reconstruction kernel, of which the first found is
used:

.. code-block:: python

   cases2016 = discover_mayo_challenge_2016_cases("Mayo2016", split='train')

Loading
-------

``load_mayo_case`` reads the projections of one dose level with the
DICOM-CT-PD reader and the vendor reconstruction with
``read_dicom_ct_images``, and returns them as a pair:

.. code-block:: python

   from diffct import load_mayo_case

   scan, images = load_mayo_case(case, dose='full')
   scan.projections.shape     # (views, channels, rows), float32
   images.volume.shape        # (slices, rows, columns), float32 Hounsfield units
   images.slice_spacing       # uniform z step in millimetres

``dose`` is ``'full'`` plus ``'low'`` for Mayo 2020 cases or ``'quarter'`` for
2016 cases. ``reference_images=False`` skips the vendor images.
``read_dicom_ct_images`` accepts any directory of one CT image series: slices
are sorted by their ImagePositionPatient z coordinate, the per-file rescale
slope/intercept is applied, and a missing or empty directory raises
``ValueError``.

Calibration
-----------

Unless the caller explicitly normalizes them, projections and reconstructions
are absolute attenuation quantities. CT-PD projections are corrected
absolute-attenuation line integrals after their per-view rescale; they are not
divided by water. Reconstructing them with millimetre geometry gives absolute
linear attenuation coefficients in inverse millimetres. Convert those values
with the scan's own water calibration:

.. code-block:: python

   from diffct import attenuation_to_hu, hu_to_attenuation

   hu = attenuation_to_hu(reconstruction, scan.water_attenuation)
   mu = hu_to_attenuation(images.volume, scan.water_attenuation)

``attenuation_to_hu`` applies ``1000 * (mu - mu_w) / mu_w``;
``hu_to_attenuation`` is its inverse. Both preserve the input type: NumPy
array, PyTorch tensor or scalar. The water attenuation differs per scanner and
protocol — 0.0192/mm for the Siemens case C002 versus 0.0186/mm for the GE
case L213 — so calibration must use the scan's own value rather than a
constant.

Some reconstruction code explicitly divides ``scan.projections`` by
``scan.water_attenuation`` to form its measurements. That normalization makes the
reconstruction dimensionless water units in which water is 1. In that idiom
the conversion is simply ``1000 * (reconstruction - 1)``; do not pass the
already normalized reconstruction to ``attenuation_to_hu``. Without that
explicit normalization, pass the absolute reconstruction and the scan
calibration to ``attenuation_to_hu`` as shown above.

Reconstruction
--------------

The scan returned by the loader is the same ``ClinicalScan`` that
:doc:`clinical_helical_example` reconstructs: pass it to
:func:`~diffct.clinical_geometry` for the measured trajectory, flying focal
spot and cylindrical detector, then reconstruct with
:class:`~diffct.Projector` and :func:`~diffct.solvers.cgls`. The vendor images
serve as the reference; their z axis can run opposite to the scanner z axis.
