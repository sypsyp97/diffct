Cone Beam Iterative Reconstruction
==================================

``examples/iterative_reconstruction.py`` reconstructs a 3D cone-beam phantom with
``diffct.Projector``. It runs on any supported trajectory, on one GPU, on several
GPUs in one process, or on several processes and nodes.

Run it
------

.. code-block:: bash

   python examples/iterative_reconstruction.py --trajectory helical
   python examples/iterative_reconstruction.py --trajectory circular --devices 0 1 2 3
   torchrun --nproc-per-node=4 examples/iterative_reconstruction.py

``--devices`` and ``torchrun`` cannot be used together. For multiple nodes, see
:doc:`multi_gpu`.

Options
-------

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--size``
     - Volume edge length in voxels (default 128).
   * - ``--views``
     - Number of views (default 360).
   * - ``--trajectory``
     - ``circular``, ``helical``, ``saddle`` or ``sinusoidal`` (default ``helical``).
   * - ``--algorithms``
     - One or more of ``cgls``, ``sirt`` and ``tv`` (default: all).
   * - ``--iterations``
     - Override the per-algorithm defaults (CGLS 30, SIRT 200, TV 200).
   * - ``--tv-weight``
     - TV penalty weight for ``tv`` (default 1.0).
   * - ``--noise``
     - Gaussian noise, relative to the largest projection (default 0.0).
   * - ``--phantom``
     - ``(size, size, size)`` ``.npy`` volume with values in [0, 1], used instead
       of the Shepp-Logan phantom.
   * - ``--figure``
     - Save central slices to this PNG. Requires Matplotlib.

Pipeline
--------

1. Project the phantom with ``Projector`` (cone beam, on the selected trajectory).
2. Reconstruct with CGLS and SIRT. Both use the projector and its adjoint.
3. Reconstruct with TV-regularized nonnegative least squares. The script uses
   Adam and autograd for this solver.
4. When the full sinogram is on one GPU, compute an FDK baseline. On non-circular
   trajectories, FDK is a heuristic baseline, not an exact reconstruction.

Notes
-----

- The iteration counts are demonstration settings, not a general convergence
  schedule. Choose a stopping rule for your data.
- The TV weight is an example setting. Choose it for your geometry, resolution,
  voxel spacing and data scale. The solver uses half the mean squared residual
  per ray and a mean TV term. Its voxel differences are not divided by voxel spacing.
- Each GPU needs a full float32 volume and workspace. Inputs of other floating-point
  types are converted to float32 internally.
- To adapt the loss for several ranks, see :doc:`multi_gpu`.

Source
------

.. literalinclude:: ../../examples/iterative_reconstruction.py
   :language: python
   :linenos:
   :caption: Iterative Reconstruction Example (3D cone beam, any trajectory)
