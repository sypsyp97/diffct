"""Global constants and configuration for DiffCT package.

This module defines core constants used throughout the DiffCT package,
including data types, CUDA thread block configurations, and numerical
precision parameters.
"""

import numpy as np
from numba import cuda

# ---------------------------------------------------------------------------
# Data Types and Numerical Constants
# ---------------------------------------------------------------------------

_DTYPE = np.float32
"""Default data type for numerical computations (numpy.float32)."""

_INF = _DTYPE(np.inf)
"""Floating-point infinity in default data type."""

_EPSILON = _DTYPE(1e-6)
"""Small epsilon value for numerical comparisons to avoid division by zero."""

# ---------------------------------------------------------------------------
# CUDA Thread Block Configurations
# ---------------------------------------------------------------------------

# CUDA thread block configurations optimized for different dimensionalities
# 2D blocks: 16x16 = 256 threads per block, optimal for 2D ray-tracing kernels
# Balances occupancy with shared memory usage for parallel/fan beam projections
_TPB_2D = (16, 16)
"""CUDA threads-per-block for 2D kernels (parallel/fan beam): (16, 16) = 256 threads."""

# 3D blocks: 8x8x8 = 512 threads per block, optimal for 3D cone beam kernels
# Smaller per-dimension size accommodates higher register usage in 3D algorithms
_TPB_3D = (8, 8, 8)
"""CUDA threads-per-block for 3D kernels (cone beam): (8, 8, 8) = 512 threads."""

# ---------------------------------------------------------------------------
# CUDA JIT Decorators
# ---------------------------------------------------------------------------

# CUDA fastmath optimization: enables aggressive floating-point optimizations
# Trades numerical precision for performance in ray-tracing calculations
# Safe for CT reconstruction where slight precision loss is acceptable for speed gains
_FASTMATH_DECORATOR = cuda.jit(cache=True, fastmath=True)
"""Numba CUDA JIT decorator with fastmath enabled for forward kernels."""
