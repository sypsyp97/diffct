Parallel Beam Filtered Backprojection (FBP)
===========================================

This page describes the maintained ``examples/analytical_reconstruction.py``
pipeline. See :doc:`api` for filter options and helper signatures.

Run it
------

.. code-block:: bash

   python examples/analytical_reconstruction.py --size 64 --window shepp-logan
   python examples/analytical_reconstruction.py --size 128 --figure analytical.png

The script runs all three beam types on one CUDA GPU and reports PSNR against
the synthetic phantom. ``--figure`` additionally requires Matplotlib.

Pipeline
--------

The maintained example uses a half-turn circular parallel-beam scan with 360
views. The detector has ``3 * size`` cells and pitch 0.5 in voxel units.

1. Generate a phantom slice and project it with ``Projector(beam="parallel")``.
2. Filter along the detector dimension (``dim=1``) with
   ``ramp_filter_1d(..., sample_spacing=pitch, pad_factor=2)``.
3. Multiply by ``angular_integration_weights(half_turn,
   redundant_full_scan=False)``.
4. Call ``parallel_weighted_backproject`` with the trajectory on the sinogram's
   CUDA device. This helper includes its analytical normalization.

The analytical weighted gather is separate from the cell-constant Siddon adjoint
used by ``Projector.backproject()``. Do not substitute the raw adjoint and add an
extra ``pi / number_of_views`` factor: the maintained pipeline already applies
angular integration weights and the helper's analytical normalization. These
CUDA gather helpers do not provide the same autograd path as ``Projector``.

Maintained source
-----------------

.. literalinclude:: ../../examples/analytical_reconstruction.py
   :language: python
   :linenos:
   :caption: Parallel FBP, fan FBP and cone FDK
