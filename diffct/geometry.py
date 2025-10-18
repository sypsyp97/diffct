"""Geometry and trajectory generation for cone beam CT.

This module provides functions for generating circular, spiral, sinusoidal, saddle,
and custom trajectories for cone beam CT scanning geometries.
"""

import math
import torch
from typing import Callable, Tuple


# ============================================================================
# Trajectory Generation Functions
# ============================================================================

def circular_trajectory_3d(n_views, sid, sdd, start_angle=0.0, end_angle=None, device='cuda', dtype=torch.float32):
    """Generate circular trajectory geometry for cone-beam CT.

    Creates source and detector position matrices for a standard circular
    orbit around the z-axis, commonly used in cone-beam CT imaging.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance (SID), in physical units.
    sdd : float
        Source-to-Detector Distance (SDD), in physical units.
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    end_angle : float, optional
        Ending angle in radians (default: 2*pi, full rotation).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 3).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 3).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 3).
    det_v_vec : torch.Tensor
        Detector v-direction unit vectors, shape (n_views, 3).

    Examples
    --------
    >>> src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
    ...     n_views=360, sid=1000.0, sdd=1500.0, device='cuda'
    ... )
    >>> print(src_pos.shape)  # (360, 3)
    """
    import math

    if end_angle is None:
        end_angle = 2 * math.pi

    # Generate angles (equivalent to linspace with endpoint=False)
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Preallocate position matrices
    src_pos = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_center = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_v_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)

    # Compute trigonometric values
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Source rotates around isocenter at distance sid
    # Convention: source at (x, y, 0), rotating in xy-plane
    src_pos[:, 0] = -sid * sin_angles  # x
    src_pos[:, 1] = sid * cos_angles   # y
    src_pos[:, 2] = 0.0                # z

    # Detector center is opposite to source at distance (sdd - sid) from isocenter
    idd = sdd - sid  # Isocenter-to-Detector Distance
    det_center[:, 0] = idd * sin_angles   # x
    det_center[:, 1] = -idd * cos_angles  # y
    det_center[:, 2] = 0.0                # z

    # Detector u-direction (tangent to rotation, in xy-plane)
    det_u_vec[:, 0] = cos_angles   # x
    det_u_vec[:, 1] = sin_angles   # y
    det_u_vec[:, 2] = 0.0          # z

    # Detector v-direction (vertical, along z-axis)
    det_v_vec[:, 0] = 0.0   # x
    det_v_vec[:, 1] = 0.0   # y
    det_v_vec[:, 2] = 1.0   # z

    return src_pos, det_center, det_u_vec, det_v_vec


def random_trajectory_3d(n_views, sid_mean, sdd_mean, sid_std=0.0, pos_std=0.0,
                         angle_std=0.0, device='cuda', dtype=torch.float32, seed=None):
    """Generate random trajectory geometry for cone-beam CT.

    Creates source and detector position matrices with random perturbations,
    useful for testing reconstruction with non-ideal trajectories or for
    simulating robotic C-arm systems with positioning uncertainties.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid_mean : float
        Mean Source-to-Isocenter Distance, in physical units.
    sdd_mean : float
        Mean Source-to-Detector Distance, in physical units.
    sid_std : float, optional
        Standard deviation for SID variations (default: 0.0).
    pos_std : float, optional
        Standard deviation for random position offsets (default: 0.0).
    angle_std : float, optional
        Standard deviation for angular perturbations in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).
    seed : int, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 3).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 3).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 3).
    det_v_vec : torch.Tensor
        Detector v-direction unit vectors, shape (n_views, 3).

    Examples
    --------
    >>> # Generate random trajectory with 10% SID variation and 5mm position noise
    >>> src_pos, det_center, det_u_vec, det_v_vec = random_trajectory_3d(
    ...     n_views=180, sid_mean=1000.0, sdd_mean=1500.0,
    ...     sid_std=100.0, pos_std=5.0, seed=42, device='cuda'
    ... )
    """
    import math

    if seed is not None:
        torch.manual_seed(seed)

    # Start with circular trajectory
    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        n_views, sid_mean, sdd_mean, device=device, dtype=dtype
    )

    # Add random perturbations
    if sid_std > 0.0:
        # Random SID variations
        sid_perturbations = torch.randn(n_views, device=device, dtype=dtype) * sid_std
        sdd_perturbations = sid_perturbations.clone()  # Keep SDD-SID distance approximately constant

        # Compute angles from existing positions
        angles = torch.atan2(-src_pos[:, 0], src_pos[:, 1])
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # Apply SID perturbations
        sid_actual = sid_mean + sid_perturbations
        src_pos[:, 0] = -sid_actual * sin_angles
        src_pos[:, 1] = sid_actual * cos_angles

        # Apply SDD perturbations
        sdd_actual = sdd_mean + sdd_perturbations
        idd = sdd_actual - sid_actual
        det_center[:, 0] = idd * sin_angles
        det_center[:, 1] = -idd * cos_angles

    if pos_std > 0.0:
        # Random 3D position offsets
        src_pos += torch.randn_like(src_pos) * pos_std
        det_center += torch.randn_like(det_center) * pos_std

    if angle_std > 0.0:
        # Random angular perturbations (rotate detector orientation)
        angle_perturbations = torch.randn(n_views, device=device, dtype=dtype) * angle_std
        cos_perturb = torch.cos(angle_perturbations)
        sin_perturb = torch.sin(angle_perturbations)

        # Rotate u and v vectors
        u_new_x = det_u_vec[:, 0] * cos_perturb - det_u_vec[:, 1] * sin_perturb
        u_new_y = det_u_vec[:, 0] * sin_perturb + det_u_vec[:, 1] * cos_perturb
        det_u_vec[:, 0] = u_new_x
        det_u_vec[:, 1] = u_new_y

    # Renormalize direction vectors to ensure they remain unit vectors
    det_u_vec = det_u_vec / torch.norm(det_u_vec, dim=1, keepdim=True)
    det_v_vec = det_v_vec / torch.norm(det_v_vec, dim=1, keepdim=True)

    return src_pos, det_center, det_u_vec, det_v_vec


def spiral_trajectory_3d(n_views, sid, sdd, z_range=100.0, n_turns=2.0,
                        start_angle=0.0, device='cuda', dtype=torch.float32):
    """Generate spiral (helical) trajectory geometry for cone-beam CT.

    Creates a helical trajectory where the source moves in a spiral pattern
    around the z-axis, commonly used in helical CT scanners.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance (SID), in physical units.
    sdd : float
        Source-to-Detector Distance (SDD), in physical units.
    z_range : float, optional
        Total z-axis range covered during the spiral (default: 100.0).
    n_turns : float, optional
        Number of complete rotations (default: 2.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 3).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 3).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 3).
    det_v_vec : torch.Tensor
        Detector v-direction unit vectors, shape (n_views, 3).

    Examples
    --------
    >>> src_pos, det_center, det_u_vec, det_v_vec = spiral_trajectory_3d(
    ...     n_views=360, sid=1000.0, sdd=1500.0, z_range=200.0, n_turns=3.0
    ... )
    """
    # Generate angles for spiral motion
    end_angle = start_angle + 2 * math.pi * n_turns
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Generate z positions (linear motion along z-axis)
    z_positions = torch.linspace(-z_range/2, z_range/2, n_views, device=device, dtype=dtype)

    # Preallocate position matrices
    src_pos = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_center = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_v_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)

    # Compute trigonometric values
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Source rotates around z-axis with z-motion
    src_pos[:, 0] = -sid * sin_angles
    src_pos[:, 1] = sid * cos_angles
    src_pos[:, 2] = z_positions

    # Detector center opposite to source
    idd = sdd - sid
    det_center[:, 0] = idd * sin_angles
    det_center[:, 1] = -idd * cos_angles
    det_center[:, 2] = z_positions

    # Detector u-direction (tangent to rotation)
    det_u_vec[:, 0] = cos_angles
    det_u_vec[:, 1] = sin_angles
    det_u_vec[:, 2] = 0.0

    # Detector v-direction (vertical)
    det_v_vec[:, 0] = 0.0
    det_v_vec[:, 1] = 0.0
    det_v_vec[:, 2] = 1.0

    return src_pos, det_center, det_u_vec, det_v_vec


def sinusoidal_trajectory_3d(n_views, sid, sdd, amplitude=50.0, frequency=2.0,
                            start_angle=0.0, device='cuda', dtype=torch.float32):
    """Generate sinusoidal trajectory geometry for cone-beam CT.

    Creates a trajectory where the source follows a sinusoidal path in 3D space,
    with radial oscillations as it rotates around the z-axis.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Mean Source-to-Isocenter Distance (SID), in physical units.
    sdd : float
        Source-to-Detector Distance (SDD), in physical units.
    amplitude : float, optional
        Amplitude of sinusoidal radial variation (default: 50.0).
    frequency : float, optional
        Number of oscillation cycles per full rotation (default: 2.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 3).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 3).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 3).
    det_v_vec : torch.Tensor
        Detector v-direction unit vectors, shape (n_views, 3).

    Examples
    --------
    >>> src_pos, det_center, det_u_vec, det_v_vec = sinusoidal_trajectory_3d(
    ...     n_views=360, sid=1000.0, sdd=1500.0, amplitude=100.0, frequency=3.0
    ... )
    """
    # Generate angles
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Compute sinusoidal radial variation
    radial_variation = amplitude * torch.sin(frequency * angles)
    sid_varying = sid + radial_variation

    # Preallocate position matrices
    src_pos = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_center = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_v_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)

    # Compute trigonometric values
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Source with sinusoidal radial variation
    src_pos[:, 0] = -sid_varying * sin_angles
    src_pos[:, 1] = sid_varying * cos_angles
    src_pos[:, 2] = 0.0

    # Detector center
    idd = sdd - sid_varying
    det_center[:, 0] = idd * sin_angles
    det_center[:, 1] = -idd * cos_angles
    det_center[:, 2] = 0.0

    # Detector u-direction
    det_u_vec[:, 0] = cos_angles
    det_u_vec[:, 1] = sin_angles
    det_u_vec[:, 2] = 0.0

    # Detector v-direction
    det_v_vec[:, 0] = 0.0
    det_v_vec[:, 1] = 0.0
    det_v_vec[:, 2] = 1.0

    return src_pos, det_center, det_u_vec, det_v_vec


def saddle_trajectory_3d(n_views, sid, sdd, z_amplitude=50.0, radial_amplitude=30.0,
                        start_angle=0.0, device='cuda', dtype=torch.float32):
    """Generate saddle-shaped trajectory geometry for cone-beam CT.

    Creates a saddle-shaped trajectory where the source moves in both z and radial
    directions following a saddle surface (hyperbolic paraboloid).

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Mean Source-to-Isocenter Distance (SID), in physical units.
    sdd : float
        Source-to-Detector Distance (SDD), in physical units.
    z_amplitude : float, optional
        Amplitude of z-axis variation (default: 50.0).
    radial_amplitude : float, optional
        Amplitude of radial variation (default: 30.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 3).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 3).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 3).
    det_v_vec : torch.Tensor
        Detector v-direction unit vectors, shape (n_views, 3).

    Examples
    --------
    >>> src_pos, det_center, det_u_vec, det_v_vec = saddle_trajectory_3d(
    ...     n_views=360, sid=1000.0, sdd=1500.0, z_amplitude=80.0, radial_amplitude=50.0
    ... )
    """
    # Generate angles
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Saddle surface: z varies with cos(2*theta), radius varies with sin(2*theta)
    z_positions = z_amplitude * torch.cos(2 * angles)
    radial_variation = radial_amplitude * torch.sin(2 * angles)
    sid_varying = sid + radial_variation

    # Preallocate position matrices
    src_pos = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_center = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_v_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)

    # Compute trigonometric values
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Source with saddle shape
    src_pos[:, 0] = -sid_varying * sin_angles
    src_pos[:, 1] = sid_varying * cos_angles
    src_pos[:, 2] = z_positions

    # Detector center
    idd = sdd - sid_varying
    det_center[:, 0] = idd * sin_angles
    det_center[:, 1] = -idd * cos_angles
    det_center[:, 2] = z_positions

    # Detector u-direction
    det_u_vec[:, 0] = cos_angles
    det_u_vec[:, 1] = sin_angles
    det_u_vec[:, 2] = 0.0

    # Detector v-direction
    det_v_vec[:, 0] = 0.0
    det_v_vec[:, 1] = 0.0
    det_v_vec[:, 2] = 1.0

    return src_pos, det_center, det_u_vec, det_v_vec


def custom_trajectory_3d(n_views, sid, sdd,
                        source_path_fn: Callable[[torch.Tensor, float], torch.Tensor],
                        start_angle=0.0, device='cuda', dtype=torch.float32):
    """Generate custom trajectory geometry for cone-beam CT using user-defined function.

    Creates a trajectory based on a user-provided function that defines the source
    position as a function of angle and SID.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance (SID), in physical units.
    sdd : float
        Source-to-Detector Distance (SDD), in physical units.
    source_path_fn : Callable[[torch.Tensor, float], torch.Tensor]
        Function that takes (angles, sid) and returns source positions (n_views, 3).
        The function signature should be: f(angles: torch.Tensor, sid: float) -> torch.Tensor
        where angles has shape (n_views,) and output has shape (n_views, 3).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 3).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 3).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 3).
    det_v_vec : torch.Tensor
        Detector v-direction unit vectors, shape (n_views, 3).

    Examples
    --------
    >>> # Define a custom figure-8 trajectory
    >>> def figure8_path(angles, sid):
    ...     src_pos = torch.zeros((len(angles), 3), device=angles.device, dtype=angles.dtype)
    ...     src_pos[:, 0] = -sid * torch.sin(angles)
    ...     src_pos[:, 1] = sid * torch.cos(angles) * torch.sin(angles)
    ...     src_pos[:, 2] = 50 * torch.sin(2 * angles)
    ...     return src_pos
    >>>
    >>> src_pos, det_center, det_u_vec, det_v_vec = custom_trajectory_3d(
    ...     n_views=360, sid=1000.0, sdd=1500.0, source_path_fn=figure8_path
    ... )
    """
    # Generate angles
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Get source positions from user-defined function
    src_pos = source_path_fn(angles, sid)

    if src_pos.shape != (n_views, 3):
        raise ValueError(f"source_path_fn must return tensor of shape ({n_views}, 3), "
                        f"got {src_pos.shape}")

    # Preallocate remaining position matrices
    det_center = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)
    det_v_vec = torch.zeros((n_views, 3), device=device, dtype=dtype)

    # For each view, compute detector position and orientation
    for i in range(n_views):
        # Vector from isocenter to source
        src_vec = src_pos[i]
        src_vec_norm = torch.norm(src_vec)
        src_unit = src_vec / src_vec_norm

        # Detector center is opposite to source
        det_center[i] = -src_unit * (sdd - src_vec_norm)

        # Detector u-direction: perpendicular to source direction in xy-plane
        # If source is along z-axis, use x-direction
        if torch.abs(src_vec[0]) < 1e-6 and torch.abs(src_vec[1]) < 1e-6:
            det_u_vec[i] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        else:
            # Perpendicular in xy-plane
            u_unnorm = torch.tensor([-src_vec[1], src_vec[0], 0.0], device=device, dtype=dtype)
            det_u_vec[i] = u_unnorm / torch.norm(u_unnorm)

        # Detector v-direction: cross product of src_unit and det_u
        det_v_vec[i] = torch.cross(src_unit, det_u_vec[i])
        det_v_vec[i] = det_v_vec[i] / torch.norm(det_v_vec[i])

    return src_pos, det_center, det_u_vec, det_v_vec


# ============================================================================
# 2D Fan Beam Trajectory Generation Functions
# ============================================================================

def circular_trajectory_2d_fan(n_views, sid, sdd, start_angle=0.0, end_angle=None, device='cuda', dtype=torch.float32):
    """Generate circular trajectory geometry for 2D fan-beam CT.

    Creates source and detector position matrices for a standard circular
    orbit around the origin, commonly used in 2D fan-beam CT imaging.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance (SID), in physical units.
    sdd : float
        Source-to-Detector Distance (SDD), in physical units.
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    end_angle : float, optional
        Ending angle in radians (default: 2*pi, full rotation).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 2).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 2).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 2).

    Examples
    --------
    >>> src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(
    ...     n_views=360, sid=1000.0, sdd=1500.0, device='cuda'
    ... )
    >>> print(src_pos.shape)  # (360, 2)
    """
    if end_angle is None:
        end_angle = 2 * math.pi

    # Generate angles
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Preallocate position matrices
    src_pos = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_center = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 2), device=device, dtype=dtype)

    # Compute trigonometric values
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Source rotates around isocenter at distance sid
    src_pos[:, 0] = -sid * sin_angles  # x
    src_pos[:, 1] = sid * cos_angles   # y

    # Detector center is opposite to source at distance (sdd - sid) from isocenter
    idd = sdd - sid  # Isocenter-to-Detector Distance
    det_center[:, 0] = idd * sin_angles   # x
    det_center[:, 1] = -idd * cos_angles  # y

    # Detector u-direction (tangent to rotation)
    det_u_vec[:, 0] = cos_angles   # x
    det_u_vec[:, 1] = sin_angles   # y

    return src_pos, det_center, det_u_vec


def sinusoidal_trajectory_2d_fan(n_views, sid, sdd, amplitude=50.0, frequency=2.0,
                                 start_angle=0.0, device='cuda', dtype=torch.float32):
    """Generate sinusoidal trajectory geometry for 2D fan-beam CT.

    Creates a trajectory where the source follows a sinusoidal path,
    with radial oscillations as it rotates around the origin.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Mean Source-to-Isocenter Distance (SID), in physical units.
    sdd : float
        Source-to-Detector Distance (SDD), in physical units.
    amplitude : float, optional
        Amplitude of sinusoidal radial variation (default: 50.0).
    frequency : float, optional
        Number of oscillation cycles per full rotation (default: 2.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 2).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 2).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 2).

    Examples
    --------
    >>> src_pos, det_center, det_u_vec = sinusoidal_trajectory_2d_fan(
    ...     n_views=360, sid=1000.0, sdd=1500.0, amplitude=100.0, frequency=3.0
    ... )
    """
    # Generate angles
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Compute sinusoidal radial variation
    radial_variation = amplitude * torch.sin(frequency * angles)
    sid_varying = sid + radial_variation

    # Preallocate position matrices
    src_pos = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_center = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 2), device=device, dtype=dtype)

    # Compute trigonometric values
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Source with sinusoidal radial variation
    src_pos[:, 0] = -sid_varying * sin_angles
    src_pos[:, 1] = sid_varying * cos_angles

    # Detector center
    idd = sdd - sid_varying
    det_center[:, 0] = idd * sin_angles
    det_center[:, 1] = -idd * cos_angles

    # Detector u-direction
    det_u_vec[:, 0] = cos_angles
    det_u_vec[:, 1] = sin_angles

    return src_pos, det_center, det_u_vec


def custom_trajectory_2d_fan(n_views, sid, sdd,
                             source_path_fn: Callable[[torch.Tensor, float], torch.Tensor],
                             start_angle=0.0, device='cuda', dtype=torch.float32):
    """Generate custom trajectory geometry for 2D fan-beam CT using user-defined function.

    Creates a trajectory based on a user-provided function that defines the source
    position as a function of angle and SID.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance (SID), in physical units.
    sdd : float
        Source-to-Detector Distance (SDD), in physical units.
    source_path_fn : Callable[[torch.Tensor, float], torch.Tensor]
        Function that takes (angles, sid) and returns source positions (n_views, 2).
        The function signature should be: f(angles: torch.Tensor, sid: float) -> torch.Tensor
        where angles has shape (n_views,) and output has shape (n_views, 2).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    src_pos : torch.Tensor
        Source positions, shape (n_views, 2).
    det_center : torch.Tensor
        Detector center positions, shape (n_views, 2).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors, shape (n_views, 2).

    Examples
    --------
    >>> # Define a custom elliptical trajectory
    >>> def ellipse_path(angles, sid):
    ...     src_pos = torch.zeros((len(angles), 2), device=angles.device, dtype=angles.dtype)
    ...     src_pos[:, 0] = -sid * 1.5 * torch.sin(angles)  # wider in x
    ...     src_pos[:, 1] = sid * torch.cos(angles)
    ...     return src_pos
    >>>
    >>> src_pos, det_center, det_u_vec = custom_trajectory_2d_fan(
    ...     n_views=360, sid=1000.0, sdd=1500.0, source_path_fn=ellipse_path
    ... )
    """
    # Generate angles
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Get source positions from user-defined function
    src_pos = source_path_fn(angles, sid)

    if src_pos.shape != (n_views, 2):
        raise ValueError(f"source_path_fn must return tensor of shape ({n_views}, 2), "
                        f"got {src_pos.shape}")

    # Preallocate remaining position matrices
    det_center = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 2), device=device, dtype=dtype)

    # For each view, compute detector position and orientation
    for i in range(n_views):
        # Vector from isocenter to source
        src_vec = src_pos[i]
        src_vec_norm = torch.norm(src_vec)
        src_unit = src_vec / src_vec_norm

        # Detector center is opposite to source
        det_center[i] = -src_unit * (sdd - src_vec_norm)

        # Detector u-direction: perpendicular to source direction (rotate 90 degrees)
        det_u_vec[i, 0] = -src_unit[1]  # perpendicular: (-y, x)
        det_u_vec[i, 1] = src_unit[0]

    return src_pos, det_center, det_u_vec


# ============================================================================
# 2D Parallel Beam Trajectory Generation Functions
# ============================================================================

def circular_trajectory_2d_parallel(n_views, detector_distance=0.0, start_angle=0.0, end_angle=None,
                                   device='cuda', dtype=torch.float32):
    """Generate circular trajectory geometry for 2D parallel-beam CT.

    Creates ray direction and detector position matrices for a standard circular
    scanning geometry, commonly used in 2D parallel-beam CT imaging.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    detector_distance : float, optional
        Distance from isocenter to detector origin (default: 0.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    end_angle : float, optional
        Ending angle in radians (default: pi, half rotation for parallel beam).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    ray_dir : torch.Tensor
        Ray direction unit vectors for each view, shape (n_views, 2).
    det_origin : torch.Tensor
        Detector origin positions for each view, shape (n_views, 2).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors for each view, shape (n_views, 2).

    Examples
    --------
    >>> ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(
    ...     n_views=180, device='cuda'
    ... )
    >>> print(ray_dir.shape)  # (180, 2)
    """
    if end_angle is None:
        end_angle = math.pi  # Half rotation for parallel beam

    # Generate angles
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Preallocate position matrices
    ray_dir = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_origin = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 2), device=device, dtype=dtype)

    # Compute trigonometric values
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Ray direction: perpendicular to detector
    ray_dir[:, 0] = cos_angles
    ray_dir[:, 1] = sin_angles

    # Detector origin: offset from isocenter perpendicular to ray direction
    det_origin[:, 0] = -detector_distance * sin_angles
    det_origin[:, 1] = detector_distance * cos_angles

    # Detector u-direction: tangent to rotation (perpendicular to ray)
    det_u_vec[:, 0] = -sin_angles
    det_u_vec[:, 1] = cos_angles

    return ray_dir, det_origin, det_u_vec


def sinusoidal_trajectory_2d_parallel(n_views, amplitude=50.0, frequency=2.0,
                                     start_angle=0.0, device='cuda', dtype=torch.float32):
    """Generate sinusoidal trajectory geometry for 2D parallel-beam CT.

    Creates a trajectory where the detector follows a sinusoidal path
    while maintaining parallel rays.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    amplitude : float, optional
        Amplitude of sinusoidal detector displacement (default: 50.0).
    frequency : float, optional
        Number of oscillation cycles per rotation (default: 2.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    ray_dir : torch.Tensor
        Ray direction unit vectors for each view, shape (n_views, 2).
    det_origin : torch.Tensor
        Detector origin positions for each view, shape (n_views, 2).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors for each view, shape (n_views, 2).

    Examples
    --------
    >>> ray_dir, det_origin, det_u_vec = sinusoidal_trajectory_2d_parallel(
    ...     n_views=180, amplitude=100.0, frequency=3.0
    ... )
    """
    # Generate angles
    end_angle = start_angle + math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Compute sinusoidal detector displacement
    detector_offset = amplitude * torch.sin(frequency * angles)

    # Preallocate position matrices
    ray_dir = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_origin = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_u_vec = torch.zeros((n_views, 2), device=device, dtype=dtype)

    # Compute trigonometric values
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Ray direction: perpendicular to detector
    ray_dir[:, 0] = cos_angles
    ray_dir[:, 1] = sin_angles

    # Detector origin with sinusoidal offset
    det_origin[:, 0] = -detector_offset * sin_angles
    det_origin[:, 1] = detector_offset * cos_angles

    # Detector u-direction: tangent to rotation
    det_u_vec[:, 0] = -sin_angles
    det_u_vec[:, 1] = cos_angles

    return ray_dir, det_origin, det_u_vec


def custom_trajectory_2d_parallel(n_views,
                                  ray_dir_fn: Callable[[torch.Tensor], torch.Tensor],
                                  det_origin_fn: Callable[[torch.Tensor], torch.Tensor],
                                  start_angle=0.0, device='cuda', dtype=torch.float32):
    """Generate custom trajectory geometry for 2D parallel-beam CT using user-defined functions.

    Creates a trajectory based on user-provided functions that define the ray
    direction and detector origin as functions of angle.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    ray_dir_fn : Callable[[torch.Tensor], torch.Tensor]
        Function that takes angles and returns ray directions (n_views, 2).
    det_origin_fn : Callable[[torch.Tensor], torch.Tensor]
        Function that takes angles and returns detector origins (n_views, 2).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    device : str or torch.device, optional
        Device for tensors (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32).

    Returns
    -------
    ray_dir : torch.Tensor
        Ray direction unit vectors for each view, shape (n_views, 2).
    det_origin : torch.Tensor
        Detector origin positions for each view, shape (n_views, 2).
    det_u_vec : torch.Tensor
        Detector u-direction unit vectors for each view, shape (n_views, 2).

    Examples
    --------
    >>> # Define custom trajectory functions
    >>> def ray_dirs(angles):
    ...     dirs = torch.zeros((len(angles), 2), device=angles.device, dtype=angles.dtype)
    ...     dirs[:, 0] = torch.cos(angles)
    ...     dirs[:, 1] = torch.sin(angles)
    ...     return dirs
    >>>
    >>> def det_origins(angles):
    ...     origins = torch.zeros((len(angles), 2), device=angles.device, dtype=angles.dtype)
    ...     return origins
    >>>
    >>> ray_dir, det_origin, det_u_vec = custom_trajectory_2d_parallel(
    ...     n_views=180, ray_dir_fn=ray_dirs, det_origin_fn=det_origins
    ... )
    """
    # Generate angles
    end_angle = start_angle + math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + torch.arange(n_views, device=device, dtype=dtype) * step

    # Get ray directions and detector origins from user-defined functions
    ray_dir = ray_dir_fn(angles)
    det_origin = det_origin_fn(angles)

    if ray_dir.shape != (n_views, 2):
        raise ValueError(f"ray_dir_fn must return tensor of shape ({n_views}, 2), "
                        f"got {ray_dir.shape}")
    if det_origin.shape != (n_views, 2):
        raise ValueError(f"det_origin_fn must return tensor of shape ({n_views}, 2), "
                        f"got {det_origin.shape}")

    # Normalize ray directions
    ray_dir = ray_dir / torch.norm(ray_dir, dim=1, keepdim=True)

    # Detector u-direction: perpendicular to ray direction (rotate 90 degrees)
    det_u_vec = torch.zeros((n_views, 2), device=device, dtype=dtype)
    det_u_vec[:, 0] = -ray_dir[:, 1]  # perpendicular: (-y, x)
    det_u_vec[:, 1] = ray_dir[:, 0]

    return ray_dir, det_origin, det_u_vec
