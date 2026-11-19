import math

import numpy as np
import pytest

from airframe.interaction import geometry as G
from airframe.interaction.geometry import RectGeom


def test_corners_axis_aligned():
    g = RectGeom(100, 50, 40, 20)
    np.testing.assert_allclose(G.corners(g), [[80, 40], [120, 40], [120, 60], [80, 60]])


def test_corners_rotated_90():
    g = RectGeom(0, 0, 40, 20, math.pi / 2)
    c = G.corners(g)
    # width now spans the y axis
    assert np.ptp(c[:, 1]) == pytest.approx(40)
    assert np.ptp(c[:, 0]) == pytest.approx(20)


def test_local_world_roundtrip():
    g = RectGeom(30, -10, 50, 70, 0.7)
    p = np.array([12.0, 34.0])
    np.testing.assert_allclose(G.to_world(g, G.to_local(g, p)), p)


def test_contains_with_rotation():
    g = RectGeom(0, 0, 100, 10, math.pi / 4)
    assert G.contains(g, (20, 20))       # along the rotated long axis
    assert not G.contains(g, (40, 0))    # would be inside if axis aligned
    assert G.contains(g, (40, 0), margin=30)


def test_rect_from_points_any_order():
    a = G.rect_from_points((10, 80), (110, 20))
    b = G.rect_from_points((110, 20), (10, 80))
    assert a == b == RectGeom(60, 50, 100, 60, 0.0)


def test_two_point_translate_only():
    g = RectGeom(100, 100, 50, 50)
    out = G.two_point_transform(g, (0, 0), (100, 0), (10, 5), (110, 5))
    assert (out.cx, out.cy, out.w, out.h) == pytest.approx((110, 105, 50, 50))
    assert out.angle == pytest.approx(0)


def test_two_point_scale_up_and_down():
    g = RectGeom(50, 0, 40, 20)
    up = G.two_point_transform(g, (0, 0), (100, 0), (0, 0), (200, 0))
    assert (up.w, up.h) == pytest.approx((80, 40))
    assert up.cx == pytest.approx(100)  # center scales about the midpoint
    down = G.two_point_transform(g, (0, 0), (100, 0), (25, 0), (75, 0))
    assert (down.w, down.h) == pytest.approx((20, 10))


def test_two_point_rotation():
    g = RectGeom(0, 0, 40, 20)
    out = G.two_point_transform(g, (-50, 0), (50, 0), (0, -50), (0, 50))
    assert out.angle == pytest.approx(math.pi / 2)
    assert (out.cx, out.cy) == pytest.approx((0, 0), abs=1e-9)


def test_two_point_respects_min_size_and_max_scale():
    g = RectGeom(0, 0, 100, 100)
    tiny = G.two_point_transform(g, (0, 0), (100, 0), (0, 0), (1, 0), min_size=30)
    assert min(tiny.w, tiny.h) == pytest.approx(30)
    huge = G.two_point_transform(g, (0, 0), (10, 0), (0, 0), (1000, 0), max_scale=3)
    assert huge.w == pytest.approx(300)


def test_two_point_snap_angle():
    g = RectGeom(0, 0, 40, 20)
    a = math.radians(3)
    out = G.two_point_transform(g, (0, 0), (100, 0), (0, 0), (100 * math.cos(a), 100 * math.sin(a)), snap=math.radians(5))
    assert out.angle == pytest.approx(0.0)


def test_resize_from_corner_keeps_opposite_fixed():
    g = RectGeom(50, 50, 100, 100)  # corners (0,0)…(100,100)
    out = G.resize_from_corner(g, 2, (150, 130))  # drag bottom-right
    np.testing.assert_allclose(G.corners(out)[0], [0, 0], atol=1e-9)
    np.testing.assert_allclose(G.corners(out)[2], [150, 130], atol=1e-9)


def test_resize_from_corner_rotated_and_min_size():
    g = RectGeom(0, 0, 100, 60, 0.5)
    tl_before = G.corners(g)[0]
    out = G.resize_from_corner(g, 2, G.corners(g)[0], min_size=20)  # collapse onto anchor
    np.testing.assert_allclose(G.corners(out)[0], tl_before, atol=1e-9)
    assert out.w == pytest.approx(20) and out.h == pytest.approx(20)
    assert out.angle == pytest.approx(0.5)


def test_snap_and_wrap_angle():
    assert G.snap_angle(math.radians(88), math.radians(4)) == pytest.approx(math.pi / 2)
    assert G.snap_angle(math.radians(80), math.radians(4)) == pytest.approx(math.radians(80))
    assert G.wrap_angle(3 * math.pi) == pytest.approx(-math.pi)


def test_lerp_geom_takes_short_way_around():
    a = RectGeom(0, 0, 10, 10, math.radians(170))
    b = RectGeom(10, 0, 20, 10, math.radians(-170))
    mid = G.lerp_geom(a, b, 0.5)
    assert abs(abs(mid.angle) - math.pi) < 1e-9
    assert (mid.cx, mid.w) == (5, 15)


def test_nearest_corner():
    g = RectGeom(50, 50, 100, 100)
    idx, d = G.nearest_corner(g, (98, 3))
    assert idx == 1 and d == pytest.approx(math.hypot(2, 3))
