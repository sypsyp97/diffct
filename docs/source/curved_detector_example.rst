Native Curved Detector
======================

``examples/curved_detector.py`` projects a phantom onto a native cylindrical cone
detector, checks the matched backprojection and computes volume and surface
gradients on a circular scan.

.. figure:: _static/gallery/curved_detector.png
   :alt: Projections of a patient volume on a cylindrical detector
   :class: example-figure

   Vendor images of TCIA case L213 (see :ref:`data-sources`) as a volume, projected
   onto a cylindrical detector centred on the source: 888 × 625 pixels, radius
   949 mm, source 541 mm from the isocentre. Made by the ``volumes`` command of
   ``docs/assets/gallery/compute.py``; the script below uses a smaller geometry.

Run it
------

Run the commands from the root of a source checkout; ``pip install diffct`` does not
install the ``examples/`` directory. See the "From source" tab in
:doc:`installation`.

.. code-block:: bash

   python examples/curved_detector.py

This small CUDA example uses the native curved operators directly, without
resampling or FDK. See :ref:`detector-surfaces` for the callback contract.

Source
------

.. literalinclude:: ../../examples/curved_detector.py
   :language: python
   :linenos:
   :caption: Native cylindrical detector operators
