"""Horizontal swipe detection on an open palm."""

from __future__ import annotations

from collections import deque


class SwipeDetector:
    """Fires +1 (right) / -1 (left) when an open palm travels far enough, fast enough.

    Guards against accidental triggers:
      * the palm must be open for every sample in the window (``active``),
      * horizontal travel ≥ ``min_distance`` within ``max_duration`` seconds,
      * vertical drift ≤ ``max_vertical_ratio`` × horizontal travel,
      * a cooldown after each swipe.
    """

    def __init__(
        self,
        min_distance: float,
        max_duration: float = 0.45,
        max_vertical_ratio: float = 0.5,
        cooldown: float = 0.9,
    ):
        self.min_distance = min_distance
        self.max_duration = max_duration
        self.max_vertical_ratio = max_vertical_ratio
        self.cooldown = cooldown
        self._samples: deque[tuple[float, float, float]] = deque()
        self._last_fire = -1e9

    def reset(self) -> None:
        self._samples.clear()

    def update(self, t: float, x: float, y: float, active: bool) -> int:
        if not active:
            self._samples.clear()
            return 0
        self._samples.append((t, x, y))
        while self._samples and t - self._samples[0][0] > self.max_duration:
            self._samples.popleft()
        if t - self._last_fire < self.cooldown or len(self._samples) < 3:
            return 0
        t0, x0, y0 = self._samples[0]
        dx, dy = x - x0, y - y0
        if abs(dx) >= self.min_distance and abs(dy) <= self.max_vertical_ratio * abs(dx):
            self._last_fire = t
            self._samples.clear()
            return 1 if dx > 0 else -1
        return 0
