Fan Beam Filtered Backprojection (FBP)
======================================

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

The maintained example uses a full-turn circular fan-beam scan with 360 views,
``sid=2.5 * size``, ``sdd=4 * size``, ``3 * size`` detector cells and pitch 0.8.
Distances and pitch use the same units as the unit-spaced image voxels.

1. Project a phantom slice with ``Projector(beam="fan")``.
2. Apply ``fan_cosine_weights`` for the detector pitch and source-detector distance.
3. Apply ``ramp_filter_1d`` along ``dim=1``, with ``sample_spacing=pitch``
   and ``pad_factor=2``.
4. Multiply by full-turn ``angular_integration_weights`` with
   ``redundant_full_scan=True``.
5. Call ``fan_weighted_backproject`` for the distance-weighted gather and
   analytical scaling.

This example is a full scan. Short-scan redundancy handling requires compatible
angles/coverage and Parker weights; it is not enabled by changing the view count.

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
