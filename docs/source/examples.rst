Examples
========

Each page has the commands, an explanation and one result figure. Figure
captions state the data and settings. The scripts are in the ``examples/``
directory of a source checkout; see the "From source" tab in :doc:`installation`.

Analytical reconstruction
-------------------------

- :doc:`Parallel-beam FBP <fbp_parallel_example>`: FBP with a choice of ramp-filter window.
- :doc:`Fan-beam FBP <fbp_fan_example>`: fan-beam FBP with cosine weights over a full turn.
- :doc:`Cone-beam FDK <fdk_cone_example>`: FDK on a circular orbit, with its cone-beam artifacts.
- :doc:`Learnable FBP <analytical_learning>`: learn the FBP filter through autograd.

Iterative reconstruction
------------------------

- :doc:`Gradient-based reconstruction <gradient_reconstruction>`: the volume is a parameter; project it, compare with the measurement, optimize.
- :doc:`CGLS, SIRT and TV <iterative_reco_cone_example>`: solvers on circular, helical, saddle or sinusoidal 3D scans.
- :doc:`Measured walnut <walnut_example>`: FDK, SIRT, CGLS and TV on 240 measured cone-beam views.

Clinical data and detectors
---------------------------

- :doc:`Clinical helical CT <clinical_helical_example>`: measured patient projections, flying focal spot, curved detector, 8 GPUs.
- :doc:`Curved detector <curved_detector_example>`: projection, matched adjoint and gradients on a native cylindrical detector.

All scripts
-----------

The scripts in the repository ``examples/`` directory show how to use
``diffct``. Run them from the repository root.

- ``quickstart.py``: ``Projector`` basics for parallel, fan and cone beams, with
  the adjoint check and gradients.
- ``curved_detector.py``: native cylindrical cone detector projection, matched
  backprojection, adjoint check and volume/surface gradients on a circular scan.
- ``chunked_reconstruction.py``: CPU-backed CGLS with automatic CUDA tiles,
  optional manual limits and measured peak allocation; see :doc:`chunking`.
- ``disk_reconstruction.py``: blockwise disk CGLS, optional native curves,
  rank-owned spatial slabs and completed-iteration checkpoints.
- ``benchmark_execution.py``: schedule and slab/block comparisons with actual
  copy payloads, pilot metadata, timing and CUDA tensor peaks.
- ``benchmark_trajectory_admission.py``: per-call trajectory validation without
  numerical projection. Use ``--backend torch --frame-device cpu`` on CPU.
- ``benchmark_surface_geometry.py``: forward, adjoint and geometry-gradient
  timings of both curved-detector interfaces on a synthetic helical scan.
- ``benchmark_gpu_overheads.py``: synthetic launch and transfer overheads of
  trajectories, detector surfaces, solvers and tiled FDK.
- ``analytical_reconstruction.py``: parallel-beam FBP, fan-beam FBP and cone-beam
  FDK, with a choice of ramp-filter window.
- ``learnable_reconstruction.py``: build and train a filter and detector weights
  for parallel/fan FBP or cone FDK using public helpers.
- ``helical_fdk.py``: approximate axial helical FDK with explicit slice windows
  and weight gradients.
- ``iterative_reconstruction.py``: 3D cone-beam CGLS, SIRT and TV reconstruction
  on circular, helical, saddle or sinusoidal trajectories.
- ``walnut_reconstruction.py``: FDK, SIRT, CGLS and TV reconstruction of a measured
  walnut scan.
- ``geometry_calibration.py``: recovers per-view angle errors and a detector shift
  from projections, using geometry gradients.
- ``benchmark_projector.py``: compares one GPU with several GPUs.
- ``benchmark_scaling.py``: measures the 512³ README workload on 1, 4 and 8 GPUs,
  with raw timings and numerical checks.
- ``plot_trajectory.py``: plots the trajectory generators (CPU only).

See ``examples/README.md`` in the repository for all launch modes.

.. _data-sources:

Data sources
------------

- **Patient data:** cases L213 (abdomen) and C002 (chest) of McCollough et al.,
  *Low Dose CT Image and Projection Data (LDCT-and-Projection-data)*, Version 7,
  The Cancer Imaging Archive, 2020, https://doi.org/10.7937/9npb-2637; dataset
  paper: Moen et al., Medical Physics, 2020, https://doi.org/10.1002/mp.14594.
  Chest and Liver subsets: `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
  The data were acquired under grants EB017095 and EB017185 (Cynthia McCollough, PI)
  from the National Institute of Biomedical Imaging and Bioengineering. "Ground
  truth" panels show the vendor images, converted from HU to water units; all
  other panels are reconstructions or simulations computed with diffct.
- **Walnut:** Meaney 2022, Zenodo 6986012, CC BY 4.0; see ``examples/data/NOTICE``.
- **Shepp-Logan phantom:** computed by ``examples/_common.py``.

The figures are made by ``docs/assets/gallery/compute.py`` and
``docs/assets/gallery/render.py``; see ``docs/assets/gallery/NOTICE``.

.. toctree::
   :hidden:
   :caption: Analytical reconstruction

   fbp_parallel_example
   fbp_fan_example
   fdk_cone_example

.. toctree::
   :hidden:
   :caption: Iterative reconstruction

   gradient_reconstruction
   iterative_reco_cone_example
   walnut_example

.. toctree::
   :hidden:
   :caption: Clinical data and detectors

   clinical_helical_example
   curved_detector_example
