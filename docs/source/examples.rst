Examples
========

Each card links to a page with the full script and an explanation. The figures
are rendered from the same pipelines; see the caption on each page.

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: gallery-grid

   .. grid-item-card:: Parallel-beam FBP
      :link: fbp_parallel_example
      :link-type: doc
      :img-top: _static/gallery/fbp_parallel.png
      :img-alt: Parallel-beam phantom, sinogram and FBP reconstruction

      FBP on a 2D phantom, with a choice of ramp-filter window.

   .. grid-item-card:: Fan-beam FBP
      :link: fbp_fan_example
      :link-type: doc
      :img-top: _static/gallery/fbp_fan.png
      :img-alt: Fan-beam phantom, sinogram and FBP reconstruction

      Fan-beam FBP with cosine weights over a full turn.

   .. grid-item-card:: Cone-beam FDK
      :link: fdk_cone_example
      :link-type: doc
      :img-top: _static/gallery/fdk_cone.png
      :img-alt: Cone-beam projection and FDK slices

      FDK on a circular cone-beam orbit.

   .. grid-item-card:: Learnable FBP
      :link: analytical_learning
      :link-type: doc
      :img-top: _static/gallery/analytical_learning.png
      :img-alt: FBP with the Ram-Lak filter and a learned filter

      Learn the FBP filter and detector weights through autograd.

   .. grid-item-card:: Parallel-beam iterative
      :link: iterative_reco_parallel_example
      :link-type: doc
      :img-top: _static/gallery/iterative_parallel.png
      :img-alt: Parallel-beam least-squares reconstruction with Adam

      Least-squares reconstruction with autograd and Adam.

   .. grid-item-card:: Fan-beam iterative
      :link: iterative_reco_fan_example
      :link-type: doc
      :img-top: _static/gallery/iterative_fan.png
      :img-alt: Fan-beam least-squares reconstruction with Adam

      The same least-squares loop for a fan beam.

   .. grid-item-card:: Cone-beam iterative
      :link: iterative_reco_cone_example
      :link-type: doc
      :img-top: _static/gallery/iterative_cone.png
      :img-alt: Coronal slices of a simulated helical scan

      CGLS, SIRT and TV on circular, helical, saddle or sinusoidal scans.

   .. grid-item-card:: Measured walnut
      :link: walnut_example
      :link-type: doc
      :img-top: _static/gallery/walnut.png
      :img-alt: FDK, SIRT, CGLS and TV slices of a measured walnut

      FDK, SIRT, CGLS and TV on 240 measured cone-beam views.

   .. grid-item-card:: Curved detector
      :link: curved_detector_example
      :link-type: doc
      :img-top: _static/gallery/curved_detector.png
      :img-alt: Projections on a native cylindrical detector

      Native cylindrical detector: projection, matched adjoint and surface
      gradients.

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

.. toctree::
   :hidden:
   :caption: Analytical Reconstruction

   fbp_parallel_example
   fbp_fan_example
   fdk_cone_example

.. toctree::
   :hidden:
   :caption: Iterative Reconstruction

   iterative_reco_parallel_example
   iterative_reco_fan_example
   iterative_reco_cone_example

.. toctree::
   :hidden:
   :caption: Measured Data and Detectors

   walnut_example
   curved_detector_example
