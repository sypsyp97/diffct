API Reference
=============

The `diffct` package is organised into focused modules that can be combined to build differentiable CT pipelines:

- ``diffct.projectors`` – PyTorch ``autograd.Function`` implementations for forward and backward projectors
- ``diffct.geometry`` – helpers to generate detector/source trajectories for 2D and 3D scans
- ``diffct.utils`` – CUDA device management and tensor bridge utilities
- ``diffct.constants`` – low-level configuration values for advanced tuning
- ``diffct.differentiable`` – deprecated compatibility shim that re-exports the public API

Core Projector Functions
------------------------

.. currentmodule:: diffct

.. autoclass:: ParallelProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ParallelBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FanProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FanBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConeProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConeBackprojectorFunction
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

Utilities
---------

.. currentmodule:: diffct.utils

.. autoclass:: DeviceManager
   :members:

.. autoclass:: TorchCUDABridge
   :members:

Additional helper functions (prefixed with an underscore) remain available for advanced integrations that require direct control over CUDA streams or interpolation buffers.

Constants
---------

.. currentmodule:: diffct.constants

.. autodata:: _DTYPE
.. autodata:: _TPB_2D
.. autodata:: _TPB_3D
.. autodata:: _FASTMATH_DECORATOR
.. autodata:: _INF
.. autodata:: _EPSILON

These values mirror the defaults used by the CUDA kernels. They are exposed for power users who need to fine-tune launch parameters or numeric tolerances; most applications should rely on the defaults.

Backward Compatibility
----------------------

.. currentmodule:: diffct.differentiable

``diffct.differentiable`` continues to expose the legacy API surface for existing code bases. New projects should import from ``diffct`` (top level) or the specific submodules shown above.

Usage Notes
-----------

- All projector operators accept tensors on CUDA devices and return results on the same device. Use ``diffct.utils.DeviceManager`` helpers when integrating into larger code bases.
- Geometry helper functions build the ``ray_dir``, ``det_origin``, and detector orientation vectors expected by the projector operators.
- Gradients flow through both forward and backward passes; set ``requires_grad=True`` on inputs that participate in optimisation loops.
- Ensure tensors are contiguous and use consistent dtype (``torch.float32``) for maximum kernel performance.
