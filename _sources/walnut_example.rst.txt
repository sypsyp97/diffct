Measured Walnut
===============

``examples/walnut_reconstruction.py`` reconstructs a real walnut from 240
measured cone-beam projections on a full circular orbit (Meaney 2022, Zenodo
6986012, CC BY 4.0; see ``examples/data/NOTICE``). The scan geometry goes to
``Projector`` as explicit per-view tensors, the same way as a calibrated
trajectory.

Run it
------

.. code-block:: bash

   python examples/walnut_reconstruction.py --figure walnut.png

The figure shows axial and coronal centre slices of a 256³ volume: FDK with a
Hann window, SIRT (200 iterations), CGLS (20) and TV-regularized least squares
(300 Adam steps, weight 0.3).

.. image:: ../assets/walnut_measured.png
   :alt: Measured walnut: FDK, SIRT, CGLS and TV reconstructions

Simulated helical scan
----------------------

``--save-volume`` stores the FDK volume scaled to [0, 1]. Use it as the ground
truth for a simulated helical scan with 720 views and 1% noise:

.. code-block:: bash

   python examples/walnut_reconstruction.py --algorithms --save-volume walnut256.npy
   python examples/iterative_reconstruction.py --size 256 --views 720 --trajectory helical \
       --noise 0.01 --phantom walnut256.npy --figure helical.png

PSNR against the walnut volume: FDK 26.67 dB, CGLS (30 iterations) 34.13 dB,
SIRT (200) 34.52 dB, TV (200, weight 1.0) 37.77 dB.

.. image:: ../assets/walnut_helical.png
   :alt: Simulated helical scan of the walnut: FDK, CGLS, SIRT and TV reconstructions

Each figure has a provenance record (command, commit, GPU, timings, PSNR and
data checksum) in ``docs/assets/walnut_measured.json`` and
``docs/assets/walnut_helical.json``.
