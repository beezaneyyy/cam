"""Automatic pinch calibration from the user's own hand.

The user holds an open hand for a moment, then pinches for a moment. The pinch
thresholds are placed between the two measured ratios, so they adapt to finger
length, camera angle and how firmly that person pinches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

OPEN_SECONDS = 2.5
PINCH_SECONDS = 2.5
SETTLE_SECONDS = 0.6  # ignore the first moments of each phase while the user reacts


def compute_thresholds(open_samples, pinch_samples) -> tuple[float, float] | None:
    """Return (pinch_on, pinch_off), or None if the two poses are not distinguishable.

    Uses robust statistics: the 75th percentile of the pinch ratios (so a few
    frames where the fingers didn't fully meet still count as pinched) and the
    25th percentile of the open ratios.
    """
    if len(open_samples) < 10 or len(pinch_samples) < 10:
        return None
    pinched = float(np.percentile(pinch_samples, 75))
    opened = float(np.percentile(open_samples, 25))
    gap = opened - pinched
    if gap < 0.25:
        return None
    on = pinched + 0.25 * gap
    off = pinched + 0.45 * gap
    on = float(np.clip(on, 0.08, 0.45))
    off = float(np.clip(max(off, on + 0.06), 0.14, 0.65))
    return round(on, 3), round(off, 3)


@dataclass
class AutoCalibration:
    started: float
    phase: str = "open"                 # open → pinch → done | failed
    open_samples: list[float] = field(default_factory=list)
    pinch_samples: list[float] = field(default_factory=list)
    result: tuple[float, float] | None = None

    def progress(self, now: float) -> float:
        dur = OPEN_SECONDS if self.phase == "open" else PINCH_SECONDS
        start = self.started if self.phase == "open" else self.started + OPEN_SECONDS
        return min(1.0, max(0.0, (now - start) / dur))

    def update(self, now: float, ratios: list[float]) -> str:
        """Feed the current ratios of visible hands. Returns the phase."""
        elapsed = now - self.started
        if self.phase == "open":
            if elapsed >= SETTLE_SECONDS:
                self.open_samples.extend(ratios)
            if elapsed >= OPEN_SECONDS:
                self.phase = "pinch"
        elif self.phase == "pinch":
            if elapsed - OPEN_SECONDS >= SETTLE_SECONDS:
                self.pinch_samples.extend(ratios)
            if elapsed >= OPEN_SECONDS + PINCH_SECONDS:
                self.result = compute_thresholds(self.open_samples, self.pinch_samples)
                self.phase = "done" if self.result else "failed"
        return self.phase
