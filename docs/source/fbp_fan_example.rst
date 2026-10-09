Fan Beam Filtered Backprojection (FBP)
======================================

This page describes the fan-beam part of ``examples/analytical_reconstruction.py``.
For filter options and helper signatures, see :doc:`api`.

Run it
------

.. code-block:: bash

   python examples/analytical_reconstruction.py --size 64 --window shepp-logan
   python examples/analytical_reconstruction.py --size 128 --figure analytical.png

The script runs parallel, fan and cone reconstruction on one CUDA GPU and prints
PSNR against the synthetic phantom. ``--figure`` requires Matplotlib.

Pipeline
--------

The example uses a full-turn circular fan-beam scan with 360 views,
``sid = 2.5 * size``, ``sdd = 4 * size``, ``3 * size`` detector cells and a
pitch of 0.8. Distances use the same units as the unit-spaced voxels.

1. Project a phantom slice with ``Projector(beam="fan")``.
2. Apply ``fan_cosine_weights`` for the detector pitch and source-detector distance.
3. Apply ``ramp_filter_1d`` along ``dim=1``, with ``sample_spacing=pitch``
   and ``pad_factor=2``.
4. Multiply by ``angular_integration_weights`` with ``redundant_full_scan=True``.
5. Call ``fan_weighted_backproject`` for the distance-weighted gather.

This example uses a full scan. Short-scan reconstruction needs Parker weights and
suitable angular coverage. Changing the view count does not enable it.

The analytical helpers are not the ``Projector.backproject()`` adjoint. Do not
use the adjoint in this pipeline, and do not add extra scale factors.

Source
------

.. literalinclude:: ../../examples/analytical_reconstruction.py
   :language: python
   :linenos:
   :caption: Parallel FBP, fan FBP and cone FDK
