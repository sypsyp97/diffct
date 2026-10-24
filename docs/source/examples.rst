Examples
========

Each card opens a page with the commands, an explanation and a result figure.
Figure captions state the data and settings. The scripts are in the ``examples/``
directory of a source checkout; see the "From source" tab in :doc:`installation`.

Analytical reconstruction
-------------------------

.. grid:: 1 2 2 4
   :gutter: 3
   :class-container: gallery-grid

   .. grid-item-card:: Parallel-beam FBP
      :link: fbp_parallel_example
      :link-type: doc
      :img-top: _static/gallery/thumb_fbp_parallel.png
      :img-alt: Parallel-beam FBP of an abdominal slice

      FBP with a choice of ramp-filter window.

   .. grid-item-card:: Fan-beam FBP
      :link: fbp_fan_example
      :link-type: doc
      :img-top: _static/gallery/thumb_fbp_fan.png
      :img-alt: Fan-beam FBP of a chest slice

      Fan-beam FBP with cosine weights over a full turn.

   .. grid-item-card:: Cone-beam FDK
      :link: fdk_cone_example
      :link-type: doc
      :img-top: _static/gallery/thumb_fdk_cone.png
      :img-alt: Coronal FDK slice of a 3D phantom

      FDK on a circular orbit, with its cone-beam artifacts.

   .. grid-item-card:: Learnable FBP
      :link: analytical_learning
      :link-type: doc
      :img-top: _static/gallery/thumb_learning.png
      :img-alt: FBP with a learned filter

      Learn the FBP filter through autograd.

Iterative reconstruction
------------------------

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: gallery-grid

   .. grid-item-card:: Gradient-based reconstruction
      :link: gradient_reconstruction
      :link-type: doc
      :img-top: _static/gallery/thumb_gradient.png
      :img-alt: Walnut reconstructed with Adam

      The volume as parameters: forward project, compare, optimize.

   .. grid-item-card:: CGLS, SIRT and TV
      :link: iterative_reco_cone_example
      :link-type: doc
      :img-top: _static/gallery/thumb_iterative_cone.png
      :img-alt: TV reconstruction of a simulated helical scan

      Solvers on circular, helical, saddle or sinusoidal 3D scans.

   .. grid-item-card:: Measured walnut
      :link: walnut_example
      :link-type: doc
      :img-top: _static/gallery/thumb_walnut.png
      :img-alt: TV reconstruction of a measured walnut

      FDK, SIRT, CGLS and TV on 240 measured cone-beam views.

Clinical data and detectors
---------------------------

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: gallery-grid

   .. grid-item-card:: Clinical helical CT
      :link: clinical_helical_example
      :link-type: doc
      :img-top: _static/gallery/thumb_helical.png
      :img-alt: Axial slice of a measured abdominal helical scan

      Measured patient projections, flying focal spot, curved detector, 8 GPUs.

   .. grid-item-card:: Curved detector
      :link: curved_detector_example
      :link-type: doc
      :img-top: _static/gallery/thumb_curved_detector.png
      :img-alt: Projection of a patient volume on a cylindrical detector

      Native cylindrical detector: projection, matched adjoint and gradients.

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
  The Cancer Imaging Archive, 2020, https://doi.org/10.7937/9npb-2637. Chest and
  Liver subsets: `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
  The figures show reconstructions and simulations computed with diffct.
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
