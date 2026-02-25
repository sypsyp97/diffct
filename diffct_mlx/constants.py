"""Global constants and configuration for DiffCT-MLX package.

This module defines core constants used throughout the DiffCT-MLX package,
including data types, Metal thread group configurations, and numerical
precision parameters.
"""

import numpy as np
import mlx.core as mx

# ---------------------------------------------------------------------------
# Data Types and Numerical Constants
# ---------------------------------------------------------------------------

_DTYPE = np.float32
"""Default data type for numerical computations (numpy.float32)."""

_MX_DTYPE = mx.float32
"""Default MLX data type for GPU computations."""

_INF = np.float32(np.inf)
"""Floating-point infinity in default data type."""

_EPSILON = np.float32(1e-6)
"""Small epsilon value for numerical comparisons to avoid division by zero."""

# ---------------------------------------------------------------------------
# Metal Thread Group Configurations
# ---------------------------------------------------------------------------

# Metal threadgroup sizes optimized for Apple Silicon GPU
# 2D threadgroups: 16x16 = 256 threads, optimal for 2D ray-tracing kernels
_TG_2D = (16, 16)
"""Metal threads-per-threadgroup for 2D kernels (parallel/fan beam): (16, 16) = 256 threads."""

# 3D threadgroups: 8x8x8 = 512 threads, optimal for 3D cone beam kernels
_TG_3D = (8, 8, 8)
"""Metal threads-per-threadgroup for 3D kernels (cone beam): (8, 8, 8) = 512 threads."""
