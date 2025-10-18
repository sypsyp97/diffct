"""Backward compatibility module for diffct.differentiable.

This module provides backward compatibility by re-exporting all functions
and classes from the refactored modular structure. New code should import
directly from the specific modules (projectors, geometry, etc.) or from
the top-level diffct package.

DEPRECATED: This module exists only for backward compatibility.
Prefer importing from 'diffct' directly:
    from diffct import ConeProjectorFunction, circular_trajectory_3d
"""

import warnings

# Show deprecation warning when this module is imported
warnings.warn(
    "Importing from 'diffct.differentiable' is deprecated. "
    "Please import from 'diffct' directly, e.g., 'from diffct import ConeProjectorFunction'",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all public API from refactored modules
from .projectors import (
    ParallelProjectorFunction,
    ParallelBackprojectorFunction,
    FanProjectorFunction,
    FanBackprojectorFunction,
    ConeProjectorFunction,
    ConeBackprojectorFunction,
)

from .geometry import (
    circular_trajectory_3d,
    random_trajectory_3d,
)

# Also export utility functions that were previously accessible
from .utils import (
    DeviceManager,
    TorchCUDABridge,
)

from .constants import (
    _DTYPE,
    _TPB_2D,
    _TPB_3D,
    _FASTMATH_DECORATOR,
    _INF,
    _EPSILON,
)

__all__ = [
    # Projector Functions
    'ParallelProjectorFunction',
    'ParallelBackprojectorFunction',
    'FanProjectorFunction',
    'FanBackprojectorFunction',
    'ConeProjectorFunction',
    'ConeBackprojectorFunction',
    # Geometry Functions
    'circular_trajectory_3d',
    'random_trajectory_3d',
    # Utility Classes
    'DeviceManager',
    'TorchCUDABridge',
    # Constants (for advanced users)
    '_DTYPE',
    '_TPB_2D',
    '_TPB_3D',
    '_FASTMATH_DECORATOR',
    '_INF',
    '_EPSILON',
]
