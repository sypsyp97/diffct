"""Small regression checks for the diagram's surface-depth clipping.
Run: python -m unittest discover -s docs/video -p test_scan_diagram.py
"""
import unittest
import numpy as np
from scan_diagram import ScanDiagram, GREEN, GREY


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.r=ScanDiagram(np.zeros((32,32,4),np.uint8),np.full((32,32),-np.inf),width=160)
        self.r.base[:]=[180,160,140]
        self.r.surface_depth[:]=0
        self.r.clear()

    def point(self,x,y,z):
        return np.array([x,y,z])@self.r.basis

    def pixel(self,x,y):
        p=self.r.project(self.point(x,y,0))
        return tuple(np.rint(p[:2]).astype(int)[::-1])

    def test_far_segment_is_hidden(self):
        self.r.line(self.point(-1,0,-1),self.point(1,0,-1),width=4)
        np.testing.assert_array_equal(self.r.rgb,self.r.base)

    def test_front_segment_remains_visible(self):
        self.r.line(self.point(-1,0,1),self.point(1,0,1),width=4)
        np.testing.assert_allclose(self.r.rgb[self.pixel(0,0)],GREEN)

    def test_ray_crossing_surface_is_clipped_at_crossing(self):
        self.r.line(self.point(-1,0,-1),self.point(1,0,1),width=4)
        np.testing.assert_allclose(self.r.rgb[self.pixel(-.5,0)],[180,160,140])
        np.testing.assert_allclose(self.r.rgb[self.pixel(.5,0)],GREEN)

    def test_tilted_detector_is_clipped_per_pixel(self):
        self.r.triangle([self.point(-1,-1,-1),self.point(1,-1,1),self.point(0,1,0)],opacity=1)
        np.testing.assert_allclose(self.r.rgb[self.pixel(-.3,-.3)],[180,160,140])
        self.assertFalse(np.array_equal(self.r.rgb[self.pixel(.3,-.3)],[180,160,140]))
        self.assertTrue(np.all(self.r.z>=0))

    def test_nearer_line_wins_independent_of_draw_order(self):
        near=[self.point(-1,0,1),self.point(1,0,1)]
        far=[self.point(-1,0,.5),self.point(1,0,.5)]
        self.r.line(*near,color=GREEN,width=4);self.r.line(*far,color=GREY,width=4)
        first=self.r.rgb[self.pixel(0,0)].copy()
        self.r.clear()
        self.r.line(*far,color=GREY,width=4);self.r.line(*near,color=GREEN,width=4)
        np.testing.assert_allclose(self.r.rgb[self.pixel(0,0)],first)
        np.testing.assert_allclose(first,GREEN)


if __name__=='__main__':unittest.main()
