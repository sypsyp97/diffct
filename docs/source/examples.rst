Examples
========

These pages explain the example scripts in ``examples/`` of the repository: what each script computes, the pipeline it uses and how to run it.

- ``analytical_reconstruction.py``: parallel-beam FBP, fan-beam FBP and cone-beam FDK, with a choice of ramp-filter window (used by the analytical pages below)
- ``iterative_reconstruction.py``: 3D cone-beam CGLS, SIRT and TV reconstruction on any trajectory (circular, helical, saddle, sinusoidal), on one GPU, several GPUs or several nodes (used by the iterative pages below)
- ``walnut_reconstruction.py``: FDK, SIRT, CGLS and TV reconstruction of a measured walnut scan
- ``quickstart.py``: ``Projector`` basics for parallel, fan and cone beams, including the adjoint check and gradients
- ``geometry_calibration.py``: recovers per-view angle errors and a detector shift from projections, using geometry gradients
- ``benchmark_projector.py``: checks correctness and measures speed of one GPU against several GPUs
- ``plot_trajectory.py``: plots the trajectory generators (CPU only)

See ``examples/README.md`` in the repository for the launch modes and measured results.

The examples are organized into two main categories:

**Analytical Reconstruction Methods**
- Filtered backprojection (FBP) algorithms for direct reconstruction
- Standard analytical approaches used in clinical and research settings

**Iterative Reconstruction Methods**  
- Gradient-based optimization approaches using differentiable operators
- Advanced reconstruction techniques with regularization capabilities

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
