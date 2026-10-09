API Reference
=============

High-level Projector
--------------------

.. currentmodule:: diffct

.. autoclass:: Projector
   :members:
   :special-members: __call__

``Projector`` takes trajectory tensors as described in :doc:`trajectories`. For
multiple GPUs and nodes, see :doc:`multi_gpu`.

Low-level Functions
-------------------

These ``autograd.Function`` wrappers take explicit trajectory tensors. Use the
``forward`` signature as the argument order for ``.apply(...)``. For new code,
prefer ``Projector``, which checks inputs and manages devices.

.. currentmodule:: diffct

.. autoclass:: ParallelProjectorFunction
   :class-doc-from: init
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ParallelBackprojectorFunction
   :class-doc-from: init
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FanProjectorFunction
   :class-doc-from: init
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FanBackprojectorFunction
   :class-doc-from: init
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConeProjectorFunction
   :class-doc-from: init
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConeBackprojectorFunction
   :class-doc-from: init
   :members:
   :undoc-members:
   :show-inheritance:

Geometry Helpers
----------------

.. currentmodule:: diffct.geometry

**3D trajectories**

.. autofunction:: circular_trajectory_3d
.. autofunction:: random_trajectory_3d
.. autofunction:: spiral_trajectory_3d
.. autofunction:: sinusoidal_trajectory_3d
.. autofunction:: saddle_trajectory_3d
.. autofunction:: custom_trajectory_3d

**2D trajectories**

.. autofunction:: circular_trajectory_2d_parallel
.. autofunction:: sinusoidal_trajectory_2d_parallel
.. autofunction:: custom_trajectory_2d_parallel
.. autofunction:: circular_trajectory_2d_fan
.. autofunction:: sinusoidal_trajectory_2d_fan
.. autofunction:: custom_trajectory_2d_fan

Analytical Reconstruction Helpers
---------------------------------

.. currentmodule:: diffct.analytical

These helpers build the pre-weights, angular weights, filter and weighted
backprojection of an analytical FBP or FDK pipeline. The weighted backprojection
helpers do not support autograd through ``Projector``. The returned images are
already scaled.

Analytical FBP and FDK assume an acquisition model. A non-circular trajectory
does not make the result exact. The helpers accept the same per-view
``(src_pos, det_center, det_u[, det_v])`` arrays as ``Projector``.

Fan and cone backprojection accept a keyword-only ``isocenter`` vector in physical
units. With ``None`` (default), it is estimated from the detector normals. Give it
explicitly for non-circular or ambiguous geometry, including a single view. The
three weighted backprojection helpers need at least two detector bins along each
interpolated axis, including the cone detector's v axis.

.. autofunction:: detector_coordinates_1d

.. autofunction:: angular_integration_weights

.. autofunction:: fan_cosine_weights

.. autofunction:: cone_cosine_weights

.. autofunction:: parker_weights

.. autofunction:: ramp_filter_1d

.. autofunction:: parallel_weighted_backproject

.. autofunction:: fan_weighted_backproject

.. autofunction:: cone_weighted_backproject

Ramp Filter Options
-------------------

``ramp_filter_1d`` is the filter used by the analytical examples. Its signature is::

    ramp_filter_1d(sinogram_tensor, dim=-1, sample_spacing=1.0,
                   pad_factor=1, window=None, use_rfft=True)

``sample_spacing``
    Physical detector-cell spacing along ``dim``, for example ``du`` for cone beam.
    The output is in physical units. Pass ``1.0`` to work in sample units.

``pad_factor``
    Zero-pads the signal to ``pad_factor * N`` samples along ``dim``. Use ``2`` for
    cone-beam FDK. Use ``4`` when objects lie close to the detector edge.

``use_rfft``
    Uses the real FFT for real inputs. Set ``False`` only to force the complex FFT
    path.

The ``window`` argument selects a window applied to the Ram-Lak ramp:

.. list-table::
   :header-rows: 1

   * - ``window``
     - Effect
   * - ``None`` or ``"ram-lak"``
     - No window. Sharpest, with the most noise.
   * - ``"cosine"``
     - Half-cosine roll-off.
   * - ``"shepp-logan"``
     - Sinc roll-off. Classical choice.
   * - ``"hamming"``
     - Milder than Hann.
   * - ``"hann"`` or ``"hanning"``
     - Strongest high-frequency suppression.
