"""Gesture engine: raw tracking results → smoothed hand states + discrete events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from airframe.config import GestureConfig
from airframe.tracking.landmarks import RawHand, TrackingResult

from . import metrics
from .filters import OneEuroFilter
from .pinch import PinchDetector, PinchTransition
from .swipe import SwipeDetector


class EventType(Enum):
    PINCH_START = "pinch_start"
    PINCH_END = "pinch_end"
    OPEN_PALM = "open_palm"
    SWIPE = "swipe"
    HAND_LOST = "hand_lost"
    PEACE = "peace"          # ✌ held long enough → switch mode


@dataclass(frozen=True)
class GestureEvent:
    type: EventType
    hand_id: int
    position: tuple[float, float] = (0.0, 0.0)
    direction: int = 0  # swipe: +1 right, -1 left


@dataclass
class HandInfo:
    """Read-only snapshot of a hand for consumers (interaction, rendering)."""

    hand_id: int
    handedness: str
    landmarks: np.ndarray          # smoothed (21, 3)
    pinch_point: np.ndarray        # smoothed (2,)
    interaction_point: np.ndarray  # pinch point frozen while the pinch is being released
    pinch_ratio: float
    pinch_progress: float
    pinching: bool
    is_open: bool
    is_fist: bool
    palm_center: np.ndarray
    stale: bool = False            # not seen in the latest tracking result
    peace_progress: float = 0.0    # 0..1 while ✌ is being held


@dataclass
class GestureFrame:
    hands: list[HandInfo] = field(default_factory=list)
    events: list[GestureEvent] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    timestamp: float = 0.0

    def hand(self, hand_id: int) -> HandInfo | None:
        for h in self.hands:
            if h.hand_id == hand_id:
                return h
        return None


class _HandState:
    OPEN_FRAMES = 3
    PEACE_SECONDS = 0.9

    def __init__(self, hand_id: int, cfg: GestureConfig, image_width: int):
        self.hand_id = hand_id
        self.point_filter = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff)
        # Landmarks are only drawn, so they can afford a bit less smoothing
        self.lm_filter = OneEuroFilter(cfg.filter_min_cutoff * 2.0, cfg.filter_beta, cfg.filter_d_cutoff)
        self.pinch = PinchDetector(cfg.pinch_on, cfg.pinch_off, cfg.pinch_on_frames, cfg.pinch_off_frames)
        self.swipe = SwipeDetector(
            cfg.swipe_min_distance * image_width,
            cfg.swipe_max_duration,
            cfg.swipe_max_vertical_ratio,
            cfg.swipe_cooldown,
        )
        self.missed = 0
        self.frames_seen = 0
        self.open_count = 0
        self.open_fired = False
        self.frozen_point: np.ndarray | None = None
        self.peace_since: float | None = None
        self.peace_fired = False
        self.info: HandInfo | None = None


class GestureEngine:
    def __init__(self, config: GestureConfig):
        self.config = config
        self._hands: dict[int, _HandState] = {}
        self._last_result_seq = -1
        self._image_width = 1

    def reconfigure(self, config: GestureConfig) -> None:
        """Apply new tolerances (from the calibration panel) to live hands."""
        self.config = config
        for st in self._hands.values():
            st.pinch.configure(config.pinch_on, config.pinch_off, config.pinch_on_frames, config.pinch_off_frames)
            for f, mult in ((st.point_filter, 1.0), (st.lm_filter, 2.0)):
                f.min_cutoff = config.filter_min_cutoff * mult
                f.beta = config.filter_beta
            st.swipe.min_distance = config.swipe_min_distance * max(1, self._image_width)

    def reset(self) -> None:
        self._hands.clear()

    def update(self, result: TrackingResult | None, now: float) -> GestureFrame:
        """Process a tracking result. Call once per *new* result.

        Passing ``None`` (or the same result twice) just returns the current state
        without events, which lets the render loop run faster than tracking.
        """
        frame = GestureFrame(timestamp=now)
        if result is None or result.frame_seq == self._last_result_seq:
            frame.hands = [st.info for st in self._hands.values() if st.info is not None]
            if result is not None:
                frame.image_size = result.image_size
            return frame
        self._last_result_seq = result.frame_seq
        frame.image_size = result.image_size
        self._image_width = result.image_size[0] or 1
        t = result.timestamp
        seen = set()
        for raw in result.hands:
            seen.add(raw.track_id)
            st = self._hands.get(raw.track_id)
            if st is None:
                st = _HandState(raw.track_id, self.config, self._image_width)
                self._hands[raw.track_id] = st
            st.missed = 0
            self._process_hand(st, raw, t, frame.events)

        for hand_id in list(self._hands):
            if hand_id in seen:
                continue
            st = self._hands[hand_id]
            st.missed += 1
            if st.info is not None:
                st.info.stale = True
            if st.missed > self.config.lost_frames:
                if st.pinch.active and st.info is not None:
                    frame.events.append(GestureEvent(EventType.PINCH_END, hand_id, _tup(st.info.interaction_point)))
                frame.events.append(GestureEvent(EventType.HAND_LOST, hand_id))
                del self._hands[hand_id]

        frame.hands = [st.info for st in self._hands.values() if st.info is not None]
        return frame

    def _process_hand(self, st: _HandState, raw: RawHand, t: float, events: list[GestureEvent]) -> None:
        cfg = self.config
        lm_raw = raw.landmarks
        ratio = metrics.pinch_ratio(lm_raw)
        st.frames_seen += 1
        # Landmarks of a hand that just entered the view are unreliable: don't let it pinch yet.
        warming_up = st.frames_seen <= cfg.pinch_warmup_frames
        transition = st.pinch.update(max(ratio, cfg.pinch_off + 0.01) if warming_up else ratio)
        st.pinch.ratio = ratio  # keep the true value for UI / calibration
        point = st.point_filter(metrics.pinch_point(lm_raw), t)
        landmarks = st.lm_filter(lm_raw, t)

        # Freeze the interaction point while fingers separate, so releasing a
        # pinch does not nudge the element that was being held.
        if st.pinch.releasing:
            if st.frozen_point is None:
                st.frozen_point = point.copy()
        else:
            st.frozen_point = None
        interaction_point = st.frozen_point if st.frozen_point is not None else point

        is_open = metrics.is_open_hand(lm_raw, cfg.open_finger_ratio, cfg.pinch_off)
        fist = metrics.is_fist(lm_raw, cfg.open_finger_ratio)
        palm = metrics.palm_center(lm_raw)

        if transition is PinchTransition.START:
            events.append(GestureEvent(EventType.PINCH_START, st.hand_id, _tup(point)))
        elif transition is PinchTransition.END:
            events.append(GestureEvent(EventType.PINCH_END, st.hand_id, _tup(interaction_point)))
            st.frozen_point = None

        if is_open and not st.pinch.active:
            st.open_count += 1
            if st.open_count >= _HandState.OPEN_FRAMES and not st.open_fired:
                st.open_fired = True
                events.append(GestureEvent(EventType.OPEN_PALM, st.hand_id, _tup(palm)))
        else:
            st.open_count = 0
            st.open_fired = False

        peace_progress = 0.0
        if not st.pinch.active and metrics.is_peace(lm_raw, cfg.open_finger_ratio):
            if st.peace_since is None:
                st.peace_since = t
            peace_progress = min(1.0, (t - st.peace_since) / _HandState.PEACE_SECONDS)
            if peace_progress >= 1.0 and not st.peace_fired:
                st.peace_fired = True
                events.append(GestureEvent(EventType.PEACE, st.hand_id, _tup(palm)))
            if st.peace_fired:
                peace_progress = 0.0
        else:
            st.peace_since = None
            st.peace_fired = False

        if cfg.swipe_enabled:
            direction = st.swipe.update(t, float(palm[0]), float(palm[1]), is_open and not st.pinch.active)
            if direction:
                events.append(GestureEvent(EventType.SWIPE, st.hand_id, _tup(palm), direction))

        st.info = HandInfo(
            hand_id=st.hand_id,
            handedness=raw.handedness,
            landmarks=landmarks,
            pinch_point=point,
            interaction_point=interaction_point,
            pinch_ratio=ratio,
            pinch_progress=st.pinch.progress,
            pinching=st.pinch.active,
            is_open=is_open,
            is_fist=fist,
            palm_center=palm,
            peace_progress=peace_progress,
        )


def _tup(p) -> tuple[float, float]:
    return (float(p[0]), float(p[1]))
