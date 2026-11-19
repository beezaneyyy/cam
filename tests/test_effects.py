import math
import time

import numpy as np
import pytest

from airframe.effects import EffectContext, apply_effect_region, default_registry, polygon_bbox
from airframe.interaction.geometry import RectGeom, corners

REGISTRY = default_registry()
ALL = REGISTRY.names
CORE = {"neon", "rgb_split", "pixelate", "glitch", "negative", "thermal", "scanlines", "edges"}
EXTRA = {"kaleidoscope", "ascii", "halftone", "pop_art", "frosted", "night_vision", "sketch", "hologram", "ripple", "vhs"}
EXPECTED = CORE | EXTRA


def test_image(h=240, w=320, seed=1):
    """Gradient + shapes: has edges, colour and texture for every effect to act on."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (x * 255 // max(w - 1, 1)).astype(np.uint8)
    img[..., 1] = (y * 255 // max(h - 1, 1)).astype(np.uint8)
    img[..., 2] = 128
    img[h // 4: h // 2, w // 4: w // 2] = (30, 200, 240)
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)
    return img


test_image.__test__ = False  # helper, not a test


def test_registry_has_all_effects_in_order():
    assert set(ALL) == EXPECTED
    assert ALL[:8] == ["neon", "rgb_split", "pixelate", "glitch", "negative", "thermal", "scanlines", "edges"]
    with pytest.raises(ValueError):
        REGISTRY.register(REGISTRY.get("neon"))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_effect_preserves_shape_dtype_and_input(name):
    roi = test_image()
    before = roi.copy()
    out = REGISTRY.get(name).apply(roi, EffectContext(time=1.3, seed=2))
    assert out.shape == roi.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(roi, before)  # input must not be mutated


@pytest.mark.parametrize("name", sorted(EXPECTED - {"neon"}))
def test_effect_changes_pixels(name):
    roi = test_image()
    out = REGISTRY.get(name).apply(roi, EffectContext(time=0.7, seed=3))
    assert np.mean(np.abs(out.astype(int) - roi.astype(int))) > 1.0


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (2, 50), (50, 2)])
def test_effect_handles_tiny_regions(name, shape):
    roi = np.full((*shape, 3), 100, np.uint8)
    out = REGISTRY.get(name).apply(roi, EffectContext())
    assert out.shape == roi.shape


def test_negative_is_exact_inversion():
    roi = test_image()
    np.testing.assert_array_equal(REGISTRY.get("negative").apply(roi, EffectContext()), 255 - roi)


def test_pixelate_creates_flat_blocks():
    out = REGISTRY.get("pixelate").apply(test_image(), EffectContext())
    # far fewer distinct colours than the noisy source
    assert len(np.unique(out.reshape(-1, 3), axis=0)) < len(np.unique(test_image().reshape(-1, 3), axis=0)) / 10


def test_glitch_is_animated_but_deterministic():
    roi = test_image()
    g = REGISTRY.get("glitch")
    a = g.apply(roi, EffectContext(time=1.0, seed=1))
    b = g.apply(roi, EffectContext(time=1.0, seed=1))
    c = g.apply(roi, EffectContext(time=2.0, seed=1))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_edges_output_mostly_dark_with_bright_contours():
    out = REGISTRY.get("edges").apply(test_image(), EffectContext())
    gray = out.mean(axis=2)
    assert np.median(gray) < 60
    assert gray.max() > 200


# ---------------------------------------------------------------- compositing
def test_axis_aligned_region_only_changes_inside():
    frame = test_image(240, 320)
    before = frame.copy()
    poly = corners(RectGeom(160, 120, 100, 80))
    assert apply_effect_region(frame, poly, REGISTRY.get("negative"), EffectContext())
    np.testing.assert_array_equal(frame[80:160, 110:210], 255 - before[80:160, 110:210])
    outside = np.ones(frame.shape[:2], bool)
    outside[80:160, 110:210] = False
    np.testing.assert_array_equal(frame[outside], before[outside])


def test_rotated_region_masks_corners_of_bbox():
    frame = test_image(240, 320)
    before = frame.copy()
    poly = corners(RectGeom(160, 120, 100, 100, math.radians(45)))
    apply_effect_region(frame, poly, REGISTRY.get("negative"), EffectContext())
    # the rotated square's center is inverted …
    np.testing.assert_array_equal(frame[120, 160], 255 - before[120, 160])
    # … but the bbox corner (outside the diamond) is untouched
    x0, y0, _, _ = polygon_bbox(poly, 320, 240)
    np.testing.assert_array_equal(frame[y0 + 2, x0 + 2], before[y0 + 2, x0 + 2])


def test_region_partially_offscreen_is_clipped():
    frame = test_image(240, 320)
    poly = corners(RectGeom(0, 0, 100, 100))
    assert apply_effect_region(frame, poly, REGISTRY.get("thermal"), EffectContext())


def test_region_fully_offscreen_is_skipped():
    frame = test_image(240, 320)
    before = frame.copy()
    poly = corners(RectGeom(-500, -500, 100, 100))
    assert not apply_effect_region(frame, poly, REGISTRY.get("thermal"), EffectContext())
    np.testing.assert_array_equal(frame, before)


def test_opacity_blends_effect():
    frame = np.full((100, 100, 3), 100, np.uint8)
    poly = corners(RectGeom(50, 50, 100, 100))
    apply_effect_region(frame, poly, REGISTRY.get("negative"), EffectContext(), opacity=0.5)
    assert abs(int(frame[50, 50, 0]) - 127) <= 1


def test_neon_decorate_draws_glow_outside_region():
    frame = np.zeros((240, 320, 3), np.uint8)
    poly = corners(RectGeom(160, 120, 100, 80))
    apply_effect_region(frame, poly, REGISTRY.get("neon"), EffectContext())
    assert frame[120, 107].max() > 0  # just outside the left edge: glow


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_effect_is_real_time_on_large_region(name):
    """Each effect on a 640×400 region must comfortably fit in a 30 FPS budget."""
    frame = test_image(720, 1280)
    poly = corners(RectGeom(640, 360, 640, 400, 0.2))
    eff = REGISTRY.get(name)
    apply_effect_region(frame, poly, eff, EffectContext())  # warm-up
    # best of several short batches: measures the effect's cost, not other load on the machine
    batches = []
    for b in range(6):
        t = time.perf_counter()
        for i in range(3):
            apply_effect_region(frame, poly, eff, EffectContext(time=(b * 3 + i) * 0.03))
        batches.append((time.perf_counter() - t) / 3 * 1000)
    per_frame_ms = min(batches)
    assert per_frame_ms < 25, f"{name}: {per_frame_ms:.1f} ms"
