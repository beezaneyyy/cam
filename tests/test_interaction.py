import math

import numpy as np
import pytest

from airframe.config import InteractionConfig
from airframe.interaction import FrameManager, InteractionController, Mode, RectGeom
from airframe.interaction.geometry import corners

from .helpers import GestureScript

EFFECTS = ["neon", "rgb_split", "pixelate", "glitch"]


def make(min_size=40.0, max_frames=12):
    cfg = InteractionConfig(min_frame_size=min_size, max_frames=max_frames, rotation_snap_degrees=0)
    mgr = FrameManager(max_frames=max_frames, min_size=min_size)
    ctl = InteractionController(mgr, cfg, EFFECTS, "neon")
    return mgr, ctl, GestureScript()


def create_frame(ctl, s, a=(200, 200), b=(500, 400), t=0.0):
    ctl.update(s.frame([("start", 1, a)]), t)
    ctl.update(s.frame([("start", 2, b)]), t + 0.05)
    ctl.update(s.frame(), t + 0.1)
    ctl.update(s.frame([("end", 1)]), t + 0.15)
    ctl.update(s.frame([("end", 2)]), t + 0.2)


# ---------------------------------------------------------------- manager
def test_manager_create_and_remove():
    mgr = FrameManager(min_size=10)
    f = mgr.create(RectGeom(0, 0, 50, 50), "neon")
    assert f is not None and len(mgr) == 1
    assert mgr.remove(f.frame_id)
    assert len(mgr) == 0
    # removal animates: still rendered until faded out
    assert len(mgr.frames) == 1
    for _ in range(60):
        mgr.update(1 / 30, 20)
    assert mgr.frames == []


def test_manager_rejects_tiny_frames():
    mgr = FrameManager(min_size=40)
    assert mgr.create(RectGeom(0, 0, 30, 100), "neon") is None


def test_manager_max_frames_drops_oldest():
    mgr = FrameManager(max_frames=2, min_size=1)
    a = mgr.create(RectGeom(0, 0, 10, 10), "neon")
    mgr.create(RectGeom(0, 0, 10, 10), "neon")
    mgr.create(RectGeom(0, 0, 10, 10), "neon")
    assert len(mgr) == 2 and mgr.get(a.frame_id) is None


def test_manager_selection_and_hit_test_top_most():
    mgr = FrameManager(min_size=1)
    a = mgr.create(RectGeom(100, 100, 100, 100), "neon")
    b = mgr.create(RectGeom(120, 120, 100, 100), "glitch")
    assert mgr.hit_test((110, 110)) is b
    mgr.bring_to_front(a.frame_id)
    assert mgr.hit_test((110, 110)) is a
    assert mgr.hit_test((500, 500)) is None
    mgr.select(b.frame_id)
    assert mgr.selected is b
    mgr.remove(b.frame_id)
    assert mgr.selected is None


def test_frame_display_eases_toward_target():
    mgr = FrameManager(min_size=1)
    f = mgr.create(RectGeom(0, 0, 100, 100), "neon")
    f.target = RectGeom(100, 0, 100, 100)
    mgr.update(1 / 60, 20)
    assert 0 < f.display.cx < 100  # moved, but not jumped
    for _ in range(120):
        mgr.update(1 / 60, 20)
    assert f.display.cx == pytest.approx(100, abs=0.5)
    assert f.opacity == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------- controller
def test_two_pinches_create_rectangle():
    mgr, ctl, s = make()
    ctl.update(s.frame([("start", 1, (200, 200))]), 0.0)
    assert ctl.mode_of(1) is Mode.ANCHOR
    ctl.update(s.frame([("start", 2, (500, 400))]), 0.05)
    st = ctl.update(s.frame(h2=(520, 420)), 0.1)
    assert st.creation == RectGeom(360, 310, 320, 220, 0.0)
    assert st.creation_valid
    ctl.update(s.frame([("end", 1)]), 0.2)
    assert len(mgr) == 1
    f = mgr.active_frames[0]
    assert (f.target.w, f.target.h) == (320, 220)
    assert mgr.selected is f and f.effect == "neon"
    assert ctl.mode_of(2) is Mode.CONSUMED  # second hand can't trigger anything
    ctl.update(s.frame([("end", 2)]), 0.3)
    assert ctl.ops == {}


def test_too_small_creation_is_discarded():
    mgr, ctl, s = make(min_size=60)
    create_frame(ctl, s, (200, 200), (230, 400))
    assert len(mgr) == 0


def test_single_hand_pinch_in_empty_space_does_not_create():
    mgr, ctl, s = make()
    ctl.update(s.frame([("start", 1, (200, 200))]), 0)
    ctl.update(s.frame(h1=(600, 500)), 0.5)
    ctl.update(s.frame([("end", 1)]), 1.0)
    assert len(mgr) == 0


def test_pinch_inside_frame_selects_and_drags():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    mgr.select(None)
    ctl.update(s.frame([("start", 1, (300, 300))]), 1.0)
    assert ctl.mode_of(1) is Mode.DRAG and mgr.selected is f and f.held
    ctl.update(s.frame(h1=(400, 250)), 1.1)
    assert (f.target.cx, f.target.cy) == pytest.approx((450, 250))
    assert (f.target.w, f.target.h) == (300, 200)  # size untouched
    ctl.update(s.frame([("end", 1)]), 1.2)
    assert not f.held


def test_tap_empty_space_clears_selection():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    assert mgr.selected is not None
    ctl.update(s.frame([("start", 1, (900, 600))]), 2.0)
    ctl.update(s.frame([("end", 1)]), 2.1)
    assert mgr.selected is None


def test_two_hands_on_frame_scale_up_and_down():
    mgr, ctl, s = make()
    create_frame(ctl, s)  # center (350,300) size 300x200
    f = mgr.active_frames[0]
    ctl.update(s.frame([("start", 1, (250, 300))]), 1.0)
    ctl.update(s.frame([("start", 2, (450, 300))]), 1.05)
    assert ctl.mode_of(1) is Mode.TRANSFORM and ctl.mode_of(2) is Mode.TRANSFORM
    ctl.update(s.frame(h1=(150, 300), h2=(550, 300)), 1.1)   # hands apart → bigger
    assert (f.target.w, f.target.h) == pytest.approx((600, 400))
    ctl.update(s.frame(h1=(300, 300), h2=(400, 300)), 1.2)   # hands together → smaller
    assert (f.target.w, f.target.h) == pytest.approx((150, 100))


def test_two_hands_rotate_frame():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    ctl.update(s.frame([("start", 1, (250, 300))]), 1.0)
    ctl.update(s.frame([("start", 2, (450, 300))]), 1.05)
    ctl.update(s.frame(h1=(350, 200), h2=(350, 400)), 1.1)
    assert f.target.angle == pytest.approx(math.pi / 2)


def test_releasing_one_hand_of_transform_continues_as_drag():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    ctl.update(s.frame([("start", 1, (250, 300))]), 1.0)
    ctl.update(s.frame([("start", 2, (450, 300))]), 1.05)
    ctl.update(s.frame([("end", 2)]), 1.1)
    assert ctl.mode_of(1) is Mode.DRAG
    ctl.update(s.frame(h1=(260, 310)), 1.2)
    assert (f.target.cx, f.target.cy) == pytest.approx((360, 310))


def test_second_hand_in_other_frame_drags_independently():
    mgr, ctl, s = make()
    create_frame(ctl, s, (100, 100), (300, 300))
    create_frame(ctl, s, (700, 100), (900, 300), t=1.0)
    a, b = mgr.active_frames
    ctl.update(s.frame([("start", 1, (200, 200))]), 2.0)
    ctl.update(s.frame([("start", 2, (800, 200))]), 2.05)
    assert ctl.mode_of(1) is Mode.DRAG and ctl.mode_of(2) is Mode.DRAG
    ctl.update(s.frame(h1=(210, 200), h2=(790, 200)), 2.1)
    assert a.target.cx == pytest.approx(210) and b.target.cx == pytest.approx(790)


def test_corner_handle_resizes_selected_frame():
    mgr, ctl, s = make()
    create_frame(ctl, s)  # (200,200)-(500,400), selected
    f = mgr.active_frames[0]
    ctl.update(s.frame([("start", 1, (502, 398))]), 1.0)  # near bottom-right
    assert ctl.mode_of(1) is Mode.RESIZE
    ctl.update(s.frame(h1=(600, 500)), 1.1)
    np.testing.assert_allclose(corners(f.target)[0], [200, 200], atol=1e-6)
    assert (f.target.w, f.target.h) == pytest.approx((400, 300))


def test_drag_into_trash_deletes_frame():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    ctl.update(s.frame([("start", 1, (350, 300))]), 1.0)
    trash = ctl.trash_center()
    st = ctl.update(s.frame(h1=tuple(trash)), 1.1)
    assert st.trash_active and st.trash_hover
    ctl.update(s.frame([("end", 1)]), 1.2)
    assert len(mgr) == 0
    assert any("REMOVED" in n.text for n in ctl.notices)


def test_hand_lost_releases_interaction():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    ctl.update(s.frame([("start", 1, (350, 300))]), 1.0)
    ctl.update(s.frame([("lost", 1)]), 1.1)
    assert ctl.ops == {} and not f.held


def test_swipe_cycles_effect_of_selected_frame():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    ctl.update(s.frame([("swipe", 3, 1)]), 1.0)
    assert f.effect == "rgb_split"
    ctl.update(s.frame([("swipe", 3, -1)]), 2.0)
    ctl.update(s.frame([("swipe", 3, -1)]), 3.0)
    assert f.effect == EFFECTS[-1]


def test_swipe_ignored_while_hand_is_interacting():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    ctl.update(s.frame([("start", 1, (350, 300))]), 1.0)
    ctl.update(s.frame([("swipe", 1, 1)]), 1.1)
    assert f.effect == "neon"


def test_effect_without_selection_sets_default_for_new_frames():
    mgr, ctl, s = make()
    ctl.set_effect("glitch")
    create_frame(ctl, s)
    assert mgr.active_frames[0].effect == "glitch"


def test_delete_selected_and_clear_all():
    mgr, ctl, s = make()
    create_frame(ctl, s, (100, 100), (300, 300))
    create_frame(ctl, s, (500, 100), (700, 300), t=1)
    assert ctl.delete_selected() and len(mgr) == 1
    ctl.clear_all()
    assert len(mgr) == 0


def test_mouse_drag_creates_frame_alone():
    mgr, ctl, _ = make()
    ctl.pointer_down((100, 100), 0.0)
    ctl.pointer_move((300, 260))
    ctl.update(None, 0.1)
    ctl.pointer_up((300, 260), 0.2)
    assert len(mgr) == 1
    assert mgr.active_frames[0].target == RectGeom(200, 180, 200, 160, 0.0)


# ---------------------------------------------------------------- phone-style zoom
def tap(ctl, s, hid, p, t):
    ctl.update(s.frame([("start", hid, p)]), t)
    ctl.update(s.frame([("end", hid)], ratios={hid: 0.4}), t + 0.1)


def run_zoom(ctl, s, hid, ratio, t0, seconds):
    t = t0
    while t < t0 + seconds:
        t += 1 / 30
        ctl.update(s.frame(ratios={hid: ratio}), t)
    return t


def test_tap_arms_zoom_and_spreading_fingers_grows_frame():
    mgr, ctl, s = make()
    create_frame(ctl, s)  # 300 × 200 at (350, 300)
    f = mgr.active_frames[0]
    tap(ctl, s, 1, (350, 300), 1.0)
    assert ctl.zoom is not None and ctl.state.zoom_frame == f.frame_id
    t = run_zoom(ctl, s, 1, 0.4, 1.1, 0.3)      # settle: reference gap = 0.4
    t = run_zoom(ctl, s, 1, 0.8, t, 0.5)        # fingers open ×2 → size × 2^1.5
    assert f.target.w == pytest.approx(300 * 2 ** 1.5, rel=0.03)
    assert f.target.h / f.target.w == pytest.approx(200 / 300)
    assert (f.target.cx, f.target.cy) == pytest.approx((350, 300))  # zooms about its centre


def test_closing_fingers_shrinks_and_full_pinch_does_not_drag():
    mgr, ctl, s = make(min_size=40)
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    tap(ctl, s, 1, (350, 300), 1.0)
    t = run_zoom(ctl, s, 1, 0.5, 1.1, 0.3)
    t = run_zoom(ctl, s, 1, 0.3, t, 0.3)
    assert f.target.w < 300
    # fingers touch: this is a pinch event, but it must not start dragging the frame
    ctl.update(s.frame([("start", 1, (350, 300))], ratios={1: 0.1}), t + 0.03)
    ctl.update(s.frame(h1=(500, 500), ratios={1: 0.1}), t + 0.06)
    assert ctl.mode_of(1) is Mode.CONSUMED
    assert (f.target.cx, f.target.cy) == pytest.approx((350, 300))
    assert min(f.target.w, f.target.h) >= 40 - 1e-6


def test_zoom_locks_after_holding_still():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    tap(ctl, s, 1, (350, 300), 1.0)
    t = run_zoom(ctl, s, 1, 0.4, 1.1, 0.3)
    t = run_zoom(ctl, s, 1, 0.6, t, 0.4)
    t = run_zoom(ctl, s, 1, 0.6, t, 1.3)       # hold still → locked
    assert ctl.zoom is None
    size = f.target.w
    assert size > 300
    run_zoom(ctl, s, 1, 1.0, t, 0.5)           # fingers move again: nothing changes
    assert f.target.w == pytest.approx(size)
    assert any("LOCKED" in n.text for n in ctl.notices)


def test_open_hand_keeps_zooming_and_other_hand_confirms():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    tap(ctl, s, 1, (350, 300), 1.0)
    run_zoom(ctl, s, 1, 0.4, 1.1, 0.4)
    ctl.update(s.frame([("open", 1)], ratios={1: 0.9}), 1.6)  # spreading wide = "open hand"
    assert ctl.zoom is not None                                 # …which must not end the zoom
    ctl.update(s.frame([("lost", 1)]), 2.0)
    assert ctl.zoom is None
    tap(ctl, s, 1, (350, 300), 3.0)
    assert ctl.zoom is not None
    ctl.update(s.frame([("start", 2, (900, 600))]), 3.3)
    assert ctl.zoom is None and ctl.mode_of(2) is Mode.ANCHOR


def test_long_pinch_still_drags_without_zoom():
    mgr, ctl, s = make()
    create_frame(ctl, s)
    ctl.update(s.frame([("start", 1, (350, 300))]), 1.0)
    ctl.update(s.frame(h1=(450, 300)), 1.5)
    ctl.update(s.frame([("end", 1)]), 1.6)
    assert ctl.zoom is None


def test_scale_frame_mouse_wheel_respects_min_size():
    mgr, ctl, s = make(min_size=60)
    create_frame(ctl, s)
    f = mgr.active_frames[0]
    ctl.scale_frame(f.frame_id, 1.1)
    assert f.target.w == pytest.approx(330)
    for _ in range(50):
        ctl.scale_frame(f.frame_id, 0.5)
    assert min(f.target.w, f.target.h) == pytest.approx(60)
