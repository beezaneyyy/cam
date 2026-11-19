import math

import numpy as np
import pytest

from airframe.config import GestureConfig
from airframe.gestures import metrics
from airframe.gestures.engine import EventType, GestureEngine
from airframe.gestures.filters import OneEuroFilter, exp_smoothing_factor
from airframe.gestures.pinch import PinchDetector, PinchTransition
from airframe.gestures.swipe import SwipeDetector
from airframe.tracking.association import HandAssociator

from .helpers import make_landmarks, raw_hand, tracking_result


# ---------------------------------------------------------------- landmark metrics
def test_distance_2d_and_3d():
    assert metrics.distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert metrics.distance((1, 2, 2), (0, 0, 0)) == pytest.approx(3.0)


def test_palm_size_matches_synthetic_hand():
    assert metrics.palm_size(make_landmarks()) == pytest.approx(100.0)


def test_pinch_ratio_open_vs_pinched():
    open_ratio = metrics.pinch_ratio(make_landmarks())
    pinched = metrics.pinch_ratio(make_landmarks(pinch=True))
    assert open_ratio > 0.8
    assert pinched < 0.1


def test_pinch_ratio_is_scale_invariant():
    near = metrics.pinch_ratio(make_landmarks(scale=2.0))
    far = metrics.pinch_ratio(make_landmarks(scale=0.5))
    assert near == pytest.approx(far, rel=1e-5)


def test_pinch_point_is_midpoint():
    lm = make_landmarks(pinch=True)
    np.testing.assert_allclose(metrics.pinch_point(lm), (lm[4, :2] + lm[8, :2]) / 2)


def test_open_hand_and_fist_classification():
    assert metrics.is_open_hand(make_landmarks())
    assert not metrics.is_open_hand(make_landmarks(pinch=True))
    assert metrics.is_fist(make_landmarks(fist=True))
    assert not metrics.is_open_hand(make_landmarks(fist=True))
    assert metrics.extended_fingers(make_landmarks()) == 4


def test_open_hand_rotation_invariant():
    lm = make_landmarks(center=(0, 0))
    a = math.radians(70)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    lm[:, :2] = lm[:, :2] @ rot.T
    assert metrics.is_open_hand(lm)


# ---------------------------------------------------------------- pinch detector
def test_pinch_requires_consecutive_frames():
    det = PinchDetector(on=0.3, off=0.45, on_frames=2, off_frames=3)
    assert det.update(0.1) is PinchTransition.NONE
    assert det.update(0.1) is PinchTransition.START
    assert det.active


def test_pinch_single_noisy_frame_does_not_trigger():
    det = PinchDetector(on=0.3, off=0.45, on_frames=2, off_frames=3)
    for r in (0.9, 0.1, 0.9, 0.1, 0.9):
        assert det.update(r) is PinchTransition.NONE
    assert not det.active


def test_pinch_hysteresis_keeps_pinch_between_thresholds():
    det = PinchDetector(on=0.3, off=0.45, on_frames=1, off_frames=1)
    assert det.update(0.2) is PinchTransition.START
    for _ in range(10):
        assert det.update(0.4) is PinchTransition.NONE  # between on and off
    assert det.active
    assert det.update(0.5) is PinchTransition.END
    assert not det.active


def test_pinch_release_debounce():
    det = PinchDetector(on=0.3, off=0.45, on_frames=1, off_frames=3)
    det.update(0.1)
    assert det.update(0.9) is PinchTransition.NONE
    assert det.update(0.1) is PinchTransition.NONE  # dip resets the counter
    assert det.update(0.9) is PinchTransition.NONE
    assert det.update(0.9) is PinchTransition.NONE
    assert det.update(0.9) is PinchTransition.END


def test_pinch_off_is_forced_above_on():
    det = PinchDetector(on=0.4, off=0.3)
    assert det.off > det.on


def test_pinch_progress_range():
    det = PinchDetector(on=0.3, off=0.45)
    det.update(2.0)
    assert det.progress == 0.0
    det.update(0.1)
    assert det.progress == 1.0


# ---------------------------------------------------------------- smoothing
def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw, out = [], []
    for i in range(300):
        x = np.array([100.0, 100.0]) + rng.normal(0, 3, 2)
        raw.append(x)
        out.append(f(x, i / 30.0))
    assert np.std(np.array(out[50:]), axis=0).max() < 0.5 * np.std(np.array(raw[50:]), axis=0).min()


def test_one_euro_follows_fast_motion_with_beta():
    slow = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    fast = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    for i in range(30):
        x = np.array([i * 20.0, 0.0])
        a = slow(x, i / 30.0)
        b = fast(x, i / 30.0)
    target = 29 * 20.0
    assert abs(target - b[0]) < abs(target - a[0])


def test_one_euro_first_value_passthrough_and_reset():
    f = OneEuroFilter()
    np.testing.assert_allclose(f([5.0, 6.0], 0.0), [5.0, 6.0])
    f.reset()
    np.testing.assert_allclose(f([1.0, 1.0], 1.0), [1.0, 1.0])


def test_one_euro_filters_landmark_arrays():
    f = OneEuroFilter()
    out = f(np.ones((21, 3)), 0.0)
    out = f(np.ones((21, 3)) * 2, 0.033)
    assert out.shape == (21, 3)
    assert np.all((out > 1) & (out < 2))


def test_exp_smoothing_factor_frame_rate_independent():
    one_step = exp_smoothing_factor(10.0, 0.1)
    two_steps = 1 - (1 - exp_smoothing_factor(10.0, 0.05)) ** 2
    assert one_step == pytest.approx(two_steps)


# ---------------------------------------------------------------- swipe
def test_swipe_detects_fast_horizontal_motion():
    s = SwipeDetector(min_distance=300, max_duration=0.5, cooldown=1.0)
    out = [s.update(i * 0.033, 100 + i * 40, 300, True) for i in range(10)]
    assert 1 in out


def test_swipe_left_direction():
    s = SwipeDetector(min_distance=300, max_duration=0.5)
    out = [s.update(i * 0.033, 900 - i * 40, 300, True) for i in range(10)]
    assert -1 in out


def test_swipe_ignores_slow_vertical_or_inactive_motion():
    slow = SwipeDetector(min_distance=300, max_duration=0.4)
    assert not any(slow.update(i * 0.1, 100 + i * 20, 300, True) for i in range(40))
    vertical = SwipeDetector(min_distance=300, max_duration=0.5)
    assert not any(vertical.update(i * 0.033, 100 + i * 40, 100 + i * 40, True) for i in range(10))
    inactive = SwipeDetector(min_distance=300, max_duration=0.5)
    assert not any(inactive.update(i * 0.033, 100 + i * 40, 300, False) for i in range(10))


def test_swipe_cooldown():
    s = SwipeDetector(min_distance=200, max_duration=0.5, cooldown=5.0)
    fired = [s.update(i * 0.033, 100 + i * 40, 300, True) for i in range(40)]
    assert sum(1 for f in fired if f) == 1


# ---------------------------------------------------------------- association
def test_association_keeps_ids_when_order_swaps():
    assoc = HandAssociator(max_distance=300)
    a = raw_hand((200, 400), handedness="Left", track_id=-1)
    b = raw_hand((900, 400), handedness="Right", track_id=-1)
    assoc.assign([a, b])
    ida, idb = a.track_id, b.track_id
    a2 = raw_hand((210, 400), handedness="Left", track_id=-1)
    b2 = raw_hand((890, 400), handedness="Right", track_id=-1)
    assoc.assign([b2, a2])  # reversed order
    assert (a2.track_id, b2.track_id) == (ida, idb)


def test_association_new_id_for_far_hand():
    assoc = HandAssociator(max_distance=100)
    a = raw_hand((200, 400), track_id=-1)
    assoc.assign([a])
    b = raw_hand((1000, 400), track_id=-1)
    assoc.assign([b])
    assert b.track_id != a.track_id


# ---------------------------------------------------------------- engine
def _engine(**kw):
    kw.setdefault("pinch_warmup_frames", 0)
    cfg = GestureConfig(pinch_on_frames=2, pinch_off_frames=2, lost_frames=2, **kw)
    return GestureEngine(cfg)


def test_engine_emits_pinch_start_and_end():
    eng = _engine()
    seq = 0
    events = []
    for pinch in [False, True, True, True, False, False, False]:
        seq += 1
        gf = eng.update(tracking_result([raw_hand(pinch=pinch)], seq, seq / 30), seq / 30)
        events += [e.type for e in gf.events]
    assert events.count(EventType.PINCH_START) == 1
    assert events.count(EventType.PINCH_END) == 1
    assert events.index(EventType.PINCH_START) < events.index(EventType.PINCH_END)


def test_engine_same_result_twice_is_idempotent():
    eng = _engine()
    r = tracking_result([raw_hand(pinch=True)], 1, 0.0)
    eng.update(r, 0.0)
    gf = eng.update(r, 0.01)
    assert gf.events == []
    assert len(gf.hands) == 1


def test_engine_hand_lost_releases_pinch():
    eng = _engine()
    for seq in range(1, 4):
        eng.update(tracking_result([raw_hand(pinch=True)], seq, seq / 30), 0)
    events = []
    for seq in range(4, 9):
        gf = eng.update(tracking_result([], seq, seq / 30), 0)
        events += [e.type for e in gf.events]
    assert EventType.PINCH_END in events and EventType.HAND_LOST in events


def test_engine_short_dropout_keeps_pinch():
    eng = _engine()
    for seq in range(1, 4):
        eng.update(tracking_result([raw_hand(pinch=True)], seq, seq / 30), 0)
    gf = eng.update(tracking_result([], 4, 4 / 30), 0)
    assert gf.events == []
    assert gf.hands[0].pinching and gf.hands[0].stale


def test_engine_open_palm_event():
    eng = _engine()
    events = []
    for seq in range(1, 6):
        gf = eng.update(tracking_result([raw_hand()], seq, seq / 30), 0)
        events += [e.type for e in gf.events]
    assert events.count(EventType.OPEN_PALM) == 1


def test_engine_swipe_event():
    eng = _engine(swipe_min_distance=0.25)
    events = []
    for seq in range(1, 15):
        x = 200 + seq * 45
        gf = eng.update(tracking_result([raw_hand((x, 400))], seq, seq / 30), 0)
        events += [e for e in gf.events if e.type is EventType.SWIPE]
    assert len(events) == 1 and events[0].direction == 1


def test_engine_reconfigure_updates_thresholds():
    eng = _engine()
    eng.update(tracking_result([raw_hand()], 1, 0.0), 0)
    new = GestureConfig(pinch_on=0.1, pinch_off=0.2)
    eng.reconfigure(new)
    st = next(iter(eng._hands.values()))
    assert st.pinch.on == 0.1 and st.pinch.off == 0.2


def test_new_hand_cannot_pinch_during_warmup():
    eng = _engine(pinch_warmup_frames=4)
    starts = []
    for seq in range(1, 9):  # hand appears already "pinched" (e.g. garbage landmarks while entering)
        gf = eng.update(tracking_result([raw_hand(pinch=True)], seq, seq / 30), 0)
        starts += [seq for e in gf.events if e.type is EventType.PINCH_START]
    assert starts and starts[0] > 4
    assert gf.hands[0].pinch_ratio < 0.1  # UI still sees the real ratio


def test_palm_size_robust_to_foreshortened_length():
    lm = make_landmarks()
    full = metrics.palm_size(lm)
    squashed = lm.copy()
    squashed[:, 1] = (squashed[:, 1] - squashed[0, 1]) * 0.4 + squashed[0, 1]  # fingers pointing at camera
    assert metrics.palm_size(squashed) > 0.7 * full


# ---------------------------------------------------------------- auto calibration
from airframe.gestures.autocalib import OPEN_SECONDS, PINCH_SECONDS, AutoCalibration, compute_thresholds  # noqa: E402


def test_compute_thresholds_between_poses():
    on, off = compute_thresholds([0.9, 1.0, 0.95] * 10, [0.12, 0.15, 0.1] * 10)
    assert 0.15 < on < off < 0.9


def test_compute_thresholds_rejects_indistinct_poses():
    assert compute_thresholds([0.4] * 30, [0.3] * 30) is None
    assert compute_thresholds([1.0] * 3, [0.1] * 3) is None


def test_auto_calibration_flow():
    cal = AutoCalibration(started=0.0)
    t = 0.0
    while t < OPEN_SECONDS + PINCH_SECONDS + 0.1:
        ratio = 1.0 if t < OPEN_SECONDS else 0.12
        phase = cal.update(t, [ratio])
        t += 1 / 30
    assert phase == "done"
    on, off = cal.result
    assert 0.12 < on < off < 1.0
