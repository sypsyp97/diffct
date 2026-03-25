import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RECO_DIR = REPO_ROOT / "diffct_mlx" / "reconstruction_algorithms"
TEST_PACKAGE = "_diffct_reco_testpkg"


def _ensure_mlx_core():
    if os.environ.get("DIFFCT_TEST_USE_REAL_MLX") == "1":
        import mlx.core as mx

        return mx

    mx_package = sys.modules.get("mlx")
    if mx_package is None:
        mx_package = types.ModuleType("mlx")
        sys.modules["mlx"] = mx_package

    mx = types.ModuleType("mlx.core")
    mx.float32 = np.float32
    mx.array = lambda value, dtype=None: np.array(value, dtype=dtype)
    mx.zeros = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    mx.ones = lambda shape, dtype=np.float32: np.ones(shape, dtype=dtype)
    mx.zeros_like = lambda value: np.zeros_like(np.asarray(value))
    mx.maximum = np.maximum
    mx.minimum = np.minimum
    mx.where = np.where
    mx.linalg = types.SimpleNamespace(norm=lambda value: np.linalg.norm(np.asarray(value)))
    mx_package.core = mx
    sys.modules["mlx.core"] = mx
    return sys.modules["mlx.core"]


def _load_reconstruction_module(module_name: str):
    _ensure_mlx_core()
    if TEST_PACKAGE not in sys.modules:
        package = types.ModuleType(TEST_PACKAGE)
        package.__path__ = [str(RECO_DIR)]
        sys.modules[TEST_PACKAGE] = package

    qualified_name = f"{TEST_PACKAGE}.{module_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]

    spec = importlib.util.spec_from_file_location(qualified_name, RECO_DIR / f"{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {qualified_name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


MX = _ensure_mlx_core()
_CORE = _load_reconstruction_module("_core")
DART = _load_reconstruction_module("dart")


def _identity_forward(volume, projection_index):
    del projection_index
    return MX.array(np.asarray(volume), dtype=getattr(volume, "dtype", None))


def _identity_back(projection, projection_index):
    del projection_index
    return MX.array(np.asarray(projection), dtype=getattr(projection, "dtype", None))


class DARTHelperTests(unittest.TestCase):
    def test_apply_detector_border_mask_uses_cached_mask_shape(self):
        value = np.arange(16, dtype=np.float32).reshape(4, 4)
        params = _CORE.ReconstructionParameters(
            volume_shape=(1, 1),
            iteration_count=1,
            detector_border_u=1,
            detector_border_v=1,
        )

        masked = _CORE.apply_detector_border_mask(value, params, fill_value=0.0)

        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 6.0, 0.0],
                [0.0, 9.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.asarray(masked), expected)

    def test_boundary_pixels_axial_connectivity_ignores_diagonal_contacts(self):
        segmented = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        axial_boundary = DART._boundary_pixels(segmented, connectivity="axial")
        full_boundary = DART._boundary_pixels(segmented, connectivity="full")

        self.assertFalse(axial_boundary[1, 1])
        self.assertTrue(full_boundary[1, 1])

    def test_smooth_boundary_voxels_only_updates_boundary_mask(self):
        image = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        boundary_mask = np.zeros_like(image, dtype=bool)
        boundary_mask[1, 1] = True

        smoothed = DART._smooth_boundary_voxels(
            image,
            boundary_mask,
            beta=0.5,
            connectivity="axial",
        )

        expected = np.array(image, copy=True)
        expected[1, 1] = 0.5
        np.testing.assert_allclose(smoothed, expected)

    def test_segment_dart_volume_applies_binary_hole_filling(self):
        volume = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        params = DART.DARTParameters(
            volume_shape=(3, 3),
            iteration_count=1,
            gray_levels=(0.0, 1.0),
            binary_fill_holes=True,
            apply_smoothing=False,
            free_pixel_probability=0.0,
        )

        segmented = DART._segment_dart_volume(
            volume,
            np.array(params.gray_levels, dtype=np.float32),
            params,
        )

        self.assertEqual(float(segmented[1, 1]), 1.0)

    def test_forward_project_views_uses_chunked_slice_helper(self):
        calls = {"single": 0, "slice": 0}

        def forward_single(volume, projection_index):
            del volume
            calls["single"] += 1
            return np.array([projection_index], dtype=np.float32)

        def forward_slice(volume, start, stop):
            del volume
            calls["slice"] += 1
            return np.arange(start, stop, dtype=np.float32)[:, None]

        forward_single.project_slice = forward_slice  # type: ignore[attr-defined]
        params = _CORE.ReconstructionParameters(
            volume_shape=(1, 1),
            iteration_count=1,
            projection_chunk_size=2,
        )

        projections = _CORE.forward_project_views(
            np.zeros((1, 1), dtype=np.float32),
            forward_single,
            5,
            params,
        )

        self.assertEqual(calls["single"], 0)
        self.assertEqual(calls["slice"], 3)
        np.testing.assert_allclose(
            np.asarray([np.asarray(projection)[0] for projection in projections]),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )


class DARTReconstructionTests(unittest.TestCase):
    def test_reconstruct_dart_restarts_from_segmented_volume(self):
        measured_projection = np.full((2, 2), 0.4, dtype=np.float32)
        params = DART.DARTParameters(
            volume_shape=(2, 2),
            iteration_count=1,
            sart_iteration_count=1,
            initial_reconstruction_sweeps=2,
            gray_levels=(0.0, 1.0),
            free_pixel_probability=1.0,
            apply_smoothing=False,
            backprojection_scale=0.5,
            raylength_thresholding=False,
            shuffle_projection_order=False,
            random_seed=0,
        )

        reconstruction = DART.reconstruct_dart(
            [measured_projection],
            _identity_forward,
            _identity_back,
            params,
            show_progress=False,
        )

        np.testing.assert_allclose(
            np.asarray(reconstruction),
            np.full((2, 2), 0.2, dtype=np.float32),
            atol=1e-6,
        )

    def test_reconstruct_dart_stops_when_segmentation_converges(self):
        measured_projection = np.zeros((2, 2), dtype=np.float32)
        executed_outer_iterations: list[int] = []

        def capture_debug_stats(stats: dict[str, float]) -> None:
            outer_iteration = int(stats["outer_iteration"])
            if outer_iteration >= 0:
                executed_outer_iterations.append(outer_iteration)

        params = DART.DARTParameters(
            volume_shape=(2, 2),
            iteration_count=5,
            sart_iteration_count=1,
            initial_reconstruction_sweeps=1,
            gray_levels=(0.0, 1.0),
            free_pixel_probability=0.0,
            apply_smoothing=False,
            convergence_epsilon=0.0,
            backprojection_scale=0.5,
            raylength_thresholding=False,
            shuffle_projection_order=False,
            sart_debug_callback=capture_debug_stats,
            random_seed=0,
        )

        reconstruction = DART.reconstruct_dart(
            [measured_projection],
            _identity_forward,
            _identity_back,
            params,
            show_progress=False,
        )

        self.assertEqual(executed_outer_iterations, [0])
        np.testing.assert_allclose(np.asarray(reconstruction), 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
