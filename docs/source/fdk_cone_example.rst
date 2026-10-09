Cone Beam Feldkamp-Davis-Kress (FDK)
====================================

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

The maintained example uses ``Scan(size, 360)`` and a circular trajectory.
Its ``examples/_common.py::fdk`` helper performs the following steps:

1. Project a 3D phantom with ``Projector`` into a ``(views, U, V)`` sinogram.
2. Apply ``cone_cosine_weights`` for the detector geometry.
3. Filter along the detector u axis (``dim=1``) with ``sample_spacing=scan.pitch``
   and ``pad_factor=2``. The last axis is v, not u.
4. Apply full-turn ``angular_integration_weights`` with
   ``redundant_full_scan=True``.
5. Call ``cone_weighted_backproject`` for the voxel-driven weighted gather and
   analytical scaling. The example clamps negative values to zero.

FDK is approximate for a circular cone-beam scan away from the central plane;
large cone angles increase artifacts. The iterative example also displays an
FDK-like baseline on non-circular scans, but this is a heuristic extension and
not an exact reconstruction for arbitrary trajectories.

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
