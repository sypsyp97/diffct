"""Numeric contracts for the GPU explainer, without hardware or timing claims.

Run: python -m unittest discover -s docs/video -p test_gpu_example.py
"""
import importlib
from pathlib import Path
import unittest

import numpy as np
import torch

from diffct import Projector


HERE = Path(__file__).parent


class GPUExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(HERE / "data2d.npz") as source:
            cls.phantom = source["phantom"].copy()
        cls.pooled = cls.phantom.reshape(64, 4, 64, 4).mean(axis=(1, 3), dtype=np.float64)
        cls.arrays = importlib.import_module("make_gpu_example").build(cls.phantom)

    def test_shapes_and_each_displayed_contribution_are_finite_and_nonzero(self):
        shapes = {"image": (64, 64), "angles": (64,), "projection": (64, 96),
                  "tile_projection": (4, 64, 96), "backprojection": (64, 64),
                  "view_backprojection": (2, 64, 64), "row_bounds": (5,)}
        for name, shape in shapes.items():
            with self.subTest(array=name):
                array = np.asarray(self.arrays[name])
                self.assertEqual(array.shape, shape)
                self.assertTrue(np.isfinite(array).all())
        for name in ("image", "projection", "backprojection"):
            with self.subTest(array=name):
                self.assertGreater(np.linalg.norm(self.arrays[name]), 0)
        for name in ("tile_projection", "view_backprojection"):
            for index, contribution in enumerate(self.arrays[name]):
                with self.subTest(array=name, contribution=index):
                    self.assertGreater(np.linalg.norm(contribution), 0)

    def test_measured_image_pooling_and_uniform_scan(self):
        np.testing.assert_allclose(self.arrays["image"], self.pooled, rtol=0, atol=1e-7)
        np.testing.assert_allclose(self.arrays["angles"], np.arange(64) * np.pi / 64,
                                   rtol=0, atol=1e-14)
        np.testing.assert_array_equal(self.arrays["row_bounds"], [0, 16, 32, 48, 64])

    def test_axis_aligned_chords_and_global_tile_positions(self):
        # At angle zero, each ray crosses one entire image row with unit weights.
        expected = np.zeros(96)
        expected[16:80] = self.pooled.sum(axis=1)
        np.testing.assert_allclose(self.arrays["projection"][0], expected, rtol=1e-7, atol=1e-7)
        for index, (start, stop) in enumerate(zip((0, 16, 32, 48), (16, 32, 48, 64))):
            expected = np.zeros(96)
            expected[16 + start:16 + stop] = self.pooled[start:stop].sum(axis=1)
            with self.subTest(tile=index):
                np.testing.assert_allclose(self.arrays["tile_projection"][index, 0], expected,
                                           rtol=1e-7, atol=1e-7)
        # The perpendicular view traverses columns in reverse detector order.
        expected = np.zeros(96)
        expected[16:80] = self.pooled.sum(axis=0)[::-1]
        np.testing.assert_allclose(self.arrays["projection"][32], expected, rtol=1e-7, atol=1e-7)

    def test_spatial_contributions_sum_to_full_projection(self):
        np.testing.assert_allclose(np.asarray(self.arrays["tile_projection"]).sum(axis=0),
                                   self.arrays["projection"], rtol=1e-11, atol=1e-10)

    def test_view_contributions_sum_to_full_adjoint(self):
        np.testing.assert_allclose(np.asarray(self.arrays["view_backprojection"]).sum(axis=0),
                                   self.arrays["backprojection"], rtol=1e-11, atol=1e-10)

    def test_every_contribution_replays_with_public_projector_geometry(self):
        angles = torch.arange(64, dtype=torch.float64) * (np.pi / 64)
        direction = torch.stack((angles.cos(), angles.sin()), dim=1)
        detector_u = torch.stack((-angles.sin(), angles.cos()), dim=1)
        operator = Projector((direction, torch.zeros_like(direction), detector_u), (64, 64), (96,),
                             beam="parallel", detector_spacing=1, voxel_spacing=1,
                             backend="torch", precision="float64")
        image = torch.from_numpy(self.pooled)
        projection = operator.project(image)
        np.testing.assert_allclose(self.arrays["projection"], projection.numpy(), rtol=1e-7, atol=1e-7)
        for index, start in enumerate(range(0, 64, 16)):
            tile = torch.zeros_like(image)
            tile[start:start + 16] = image[start:start + 16]
            with self.subTest(tile=index):
                np.testing.assert_allclose(self.arrays["tile_projection"][index],
                                           operator.project(tile).numpy(), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(self.arrays["backprojection"], operator.backproject(projection).numpy(),
                                   rtol=1e-7, atol=1e-5)
        for index, start in enumerate((0, 32)):
            views = torch.zeros_like(projection)
            views[start:start + 32] = projection[start:start + 32]
            with self.subTest(view_half=index):
                np.testing.assert_allclose(self.arrays["view_backprojection"][index],
                                           operator.backproject(views).numpy(), rtol=1e-7, atol=1e-5)

    def test_saved_assets_reproduce_generator_output(self):
        path = HERE / "gpu_example.npz"
        self.assertTrue(path.is_file(), "gpu_example.npz is required by rendering")
        with np.load(path) as saved:
            for name, array in self.arrays.items():
                with self.subTest(array=name):
                    np.testing.assert_allclose(saved[name], array, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
