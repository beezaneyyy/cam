"""Synthetic hand landmarks for tests (no camera, no model)."""

from __future__ import annotations

import numpy as np

from airframe.gestures.engine import EventType, GestureEvent, GestureFrame, HandInfo
from airframe.tracking.landmarks import RawHand, TrackingResult

# Open right hand, palm facing camera, fingers up. Wrist at origin, palm size 100 px.
_OPEN = np.array(
    [
        [0, 0],          # 0 wrist
        [-30, -30],      # 1 thumb cmc
        [-55, -55],      # 2 thumb mcp
        [-75, -85],      # 3 thumb ip
        [-95, -110],     # 4 thumb tip
        [-30, -95],      # 5 index mcp
        [-30, -140],     # 6 index pip
        [-30, -165],     # 7 index dip
        [-30, -185],     # 8 index tip
        [0, -100],       # 9 middle mcp
        [0, -145],       # 10
        [0, -172],       # 11
        [0, -195],       # 12 middle tip
        [25, -95],       # 13 ring mcp
        [25, -138],
        [25, -162],
        [25, -180],      # 16 ring tip
        [45, -85],       # 17 pinky mcp
        [45, -120],
        [45, -140],
        [45, -155],      # 20 pinky tip
    ],
    dtype=np.float32,
)


def make_landmarks(center=(400.0, 400.0), pinch: bool = False, fist: bool = False, scale: float = 1.0) -> np.ndarray:
    lm = _OPEN.copy()
    if fist:
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            lm[tip] = lm[pip] * 0.75   # tip curled back toward the palm
        lm[4] = lm[5] + np.array([5, 10])
    if pinch:
        lm[4] = lm[8] + np.array([4.0, 2.0])
    lm = lm * scale + np.asarray(center, dtype=np.float32)
    out = np.zeros((21, 3), dtype=np.float32)
    out[:, :2] = lm
    return out


def raw_hand(center=(400, 400), pinch=False, fist=False, handedness="Right", track_id=1, scale=1.0) -> RawHand:
    return RawHand(make_landmarks(center, pinch, fist, scale), handedness, 0.99, track_id)


def tracking_result(hands, seq: int, t: float, size=(1280, 720)) -> TrackingResult:
    return TrackingResult(list(hands), seq, t, size, 10.0)


def pinch_point_of(center, scale=1.0) -> np.ndarray:
    lm = make_landmarks(center, pinch=True, scale=scale)
    return (lm[4, :2] + lm[8, :2]) / 2


def hand_info(hand_id: int, point, pinching=True) -> HandInfo:
    p = np.asarray(point, dtype=np.float64)
    return HandInfo(
        hand_id=hand_id,
        handedness="Right",
        landmarks=np.zeros((21, 3)),
        pinch_point=p,
        interaction_point=p,
        pinch_ratio=0.1 if pinching else 1.0,
        pinch_progress=1.0 if pinching else 0.0,
        pinching=pinching,
        is_open=not pinching,
        is_fist=False,
        palm_center=p,
    )


class GestureScript:
    """Builds GestureFrames by hand for controller tests."""

    def __init__(self, size=(1280, 720)):
        self.size = size
        self.points: dict[int, np.ndarray] = {}
        self.ratios: dict[int, float] = {}

    def frame(self, events=(), ratios: dict[int, float] | None = None, **moves) -> GestureFrame:
        """``ratios`` sets the thumb–index pinch ratio of hands (persisted across frames)."""
        self.ratios.update(ratios or {})
        for key, p in moves.items():
            self.points[int(key[1:])] = np.asarray(p, dtype=np.float64)
        evs = []
        for kind, hid, *rest in events:
            if kind == "start":
                self.points[hid] = np.asarray(rest[0], dtype=np.float64)
                evs.append(GestureEvent(EventType.PINCH_START, hid, tuple(rest[0])))
            elif kind == "end":
                evs.append(GestureEvent(EventType.PINCH_END, hid, tuple(self.points[hid])))
            elif kind == "swipe":
                evs.append(GestureEvent(EventType.SWIPE, hid, (0, 0), rest[0]))
            elif kind == "lost":
                evs.append(GestureEvent(EventType.HAND_LOST, hid))
                self.points.pop(hid, None)
            elif kind == "open":
                evs.append(GestureEvent(EventType.OPEN_PALM, hid, tuple(self.points.get(hid, (0, 0)))))
        hands = []
        for hid, p in self.points.items():
            h = hand_info(hid, p)
            if hid in self.ratios:
                h.pinch_ratio = self.ratios[hid]
            hands.append(h)
        return GestureFrame(hands, evs, self.size, 0.0)
