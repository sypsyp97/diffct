API Reference
=============

High-level Projector
--------------------

.. currentmodule:: diffct

.. autosummary::
   :toctree: generated
   :template: api_class.rst
   :nosignatures:

   Projector

``Projector`` takes trajectory tensors as described in :doc:`trajectories`. For
multiple GPUs and nodes, see :doc:`multi_gpu`.
Each operation also accepts ``trajectory=...`` to replace the full trajectory
for that call, preserving its view count and the constructor default.

``detector_surface`` defines native pixel positions for both projection and
matched backprojection. See :ref:`detector-surfaces` for its coordinate and
shape contract, and :doc:`examples` for a runnable cylindrical detector.

Native CPU data uses streamed CUDA execution; ``backend="torch"`` computes
on its PyTorch device. ``precision`` selects fast FP32 or strict FP32/FP64.
Leading batch/channel axes and per-axis voxel spacing are supported; see
:doc:`execution`. ``volume_chunk_shape``,
``view_chunk_size`` and ``detector_chunk_shape`` specify optional limits;
``schedule`` allows operation-specific comparisons. ``partition="space"``
assigns rank-local volume slabs. See :doc:`chunking` for memory, storage,
geometry and autograd requirements, and :doc:`multi_gpu` for ownership.

Block stores and batched surfaces
------------------------------------------------------------

.. autosummary::
   :toctree: generated
   :template: api_class.rst
   :nosignatures:

   TensorStore
   NpyStore
   ParameterizedSurface

``project_into`` and ``backproject_into`` use caller-owned numerical outputs.
``ParameterizedSurface`` exposes mutable geometry as explicit tensor
parameters and samples bounded global view/pixel rectangles. Legacy two-argument
surface callbacks remain available. See :ref:`detector-surfaces`.

Low-level Functions
-------------------

These ``autograd.Function`` wrappers take explicit trajectory tensors. Use the
``forward`` signature as the argument order for ``.apply(...)``. For new code,
prefer ``Projector``, which checks inputs and manages devices.

.. currentmodule:: diffct

.. autosummary::
   :toctree: generated
   :template: api_autograd_function.rst
   :nosignatures:

   ParallelProjectorFunction
   ParallelBackprojectorFunction
   FanProjectorFunction
   FanBackprojectorFunction
   ConeProjectorFunction
   ConeBackprojectorFunction

Geometry Helpers
----------------

.. currentmodule:: diffct.geometry

**3D trajectories**

.. autosummary::
   :toctree: generated
   :template: api_function.rst
   :nosignatures:

   circular_trajectory_3d
   circular_trajectory_3d_parallel
   random_trajectory_3d
   spiral_trajectory_3d
   sinusoidal_trajectory_3d
   saddle_trajectory_3d
   custom_trajectory_3d

**2D trajectories**

.. autosummary::
   :toctree: generated
   :template: api_function.rst
   :nosignatures:

   circular_trajectory_2d_parallel
   sinusoidal_trajectory_2d_parallel
   custom_trajectory_2d_parallel
   circular_trajectory_2d_fan
   sinusoidal_trajectory_2d_fan
   custom_trajectory_2d_fan

**Rigid transforms**

.. autosummary::
   :toctree: generated
   :template: api_function.rst
   :nosignatures:

   transform_trajectory

Analytical Reconstruction Helpers
---------------------------------

.. currentmodule:: diffct.analytical

These helpers build the pre-weights, angular weights, filter and weighted
backprojection of an analytical FBP or FDK pipeline. The weighted backprojection
helpers support sinogram autograd with dedicated transpose kernels, including
higher derivatives. They use fixed geometry and reject geometry/isocenter
gradients. The returned images are already scaled. See :doc:`analytical_learning`
for complete APIs and learnable filter/weight examples.

Analytical FBP and FDK assume a flat detector and an acquisition model. These
helpers do not accept ``detector_surface``. Native curved data must be resampled
along matching rays onto a covered virtual flat detector before using these
flat-detector helpers; resampling adds interpolation error. The native
``Projector`` forward/adjoint pair needs no such conversion. A non-circular
trajectory does not make the result exact. The helpers accept the same per-view
``(src_pos, det_center, det_u[, det_v])`` arrays as ``Projector``.

Fan and cone backprojection accept a keyword-only ``isocenter`` vector in physical
units. With ``None`` (default), it is estimated from the detector normals. Give it
explicitly for non-circular or ambiguous geometry, including a single view. The
three weighted backprojection helpers need at least two detector bins along each
interpolated axis, including the cone detector's v axis.

.. autosummary::
   :toctree: generated
   :template: api_function.rst
   :nosignatures:

   detector_coordinates_1d
   angular_integration_weights
   fan_cosine_weights
   cone_cosine_weights
   parker_weights
   ramp_filter_1d
   parallel_weighted_backproject
   fan_weighted_backproject
   cone_weighted_backproject

Complete Reconstruction Pipelines
------------------------------------

.. currentmodule:: diffct.reconstruction

.. autosummary::
   :toctree: generated
   :template: api_function.rst
   :nosignatures:

   fbp
   fdk
   fbp_into
   fdk_into
   helical_fdk
   helical_view_weights
   helical_weighted_backproject

Numerical Solvers
-----------------

.. currentmodule:: diffct.solvers

.. autosummary::
   :toctree: generated
   :template: api_class.rst
   :nosignatures:

   SolverResult

.. autosummary::
   :toctree: generated
   :template: api_function.rst
   :nosignatures:

   cgls
   sirt
   tv_reconstruction
   cgls_into
   total_variation

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
