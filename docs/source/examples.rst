Examples
========

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
- ``analytical_reconstruction.py``: parallel-beam FBP, fan-beam FBP and cone-beam
  FDK, with a choice of ramp-filter window.
- ``iterative_reconstruction.py``: 3D cone-beam CGLS, SIRT and TV reconstruction
  on circular, helical, saddle or sinusoidal trajectories.
- ``walnut_reconstruction.py``: FDK, SIRT, CGLS and TV reconstruction of a measured
  walnut scan.
- ``geometry_calibration.py``: recovers per-view angle errors and a detector shift
  from projections, using geometry gradients.
- ``benchmark_projector.py``: compares one GPU with several GPUs.
- ``plot_trajectory.py``: plots the trajectory generators (CPU only).

See ``examples/README.md`` in the repository for all launch modes.

.. toctree::
   :maxdepth: 1
   :caption: Analytical Reconstruction

   fbp_parallel_example
   fbp_fan_example
   fdk_cone_example

.. toctree::
   :maxdepth: 1
   :caption: Iterative Reconstruction

   iterative_reco_parallel_example
   iterative_reco_fan_example
   iterative_reco_cone_example

.. toctree::
   :maxdepth: 1
   :caption: Measured Data

   walnut_example

Native curved detector
----------------------

.. code-block:: bash

   python examples/curved_detector.py

This small CUDA example uses the native curved operators directly, without
resampling or FDK. See :ref:`detector-surfaces` for the callback contract.

.. literalinclude:: ../../examples/curved_detector.py
   :language: python
   :linenos:
   :caption: Native cylindrical detector operators
