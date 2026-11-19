"""Pinch state machine with hysteresis and debounce."""

from __future__ import annotations

from enum import Enum


class PinchTransition(Enum):
    NONE = 0
    START = 1
    END = 2


class PinchDetector:
    """Turns a noisy pinch ratio into clean start/end transitions.

    * Hysteresis: engage under ``on``, release only above ``off`` (> ``on``).
    * Debounce: the condition must hold for N consecutive samples, so a single
      noisy frame never starts or ends an interaction.
    """

    OPEN_REFERENCE = 0.9  # ratio of a relaxed open hand, used for the progress meter

    def __init__(self, on: float = 0.28, off: float = 0.42, on_frames: int = 2, off_frames: int = 3):
        self.configure(on, off, on_frames, off_frames)
        self.active = False
        self._count = 0
        self.ratio = 1.0

    def configure(self, on: float, off: float, on_frames: int, off_frames: int) -> None:
        if off <= on:
            off = on + 0.02
        self.on, self.off = on, off
        self.on_frames, self.off_frames = max(1, on_frames), max(1, off_frames)

    def reset(self) -> None:
        self.active = False
        self._count = 0

    @property
    def progress(self) -> float:
        """0 → hand open, 1 → pinch threshold reached. Used for UI feedback."""
        span = self.OPEN_REFERENCE - self.on
        if span <= 0:
            return 1.0 if self.active else 0.0
        return float(min(1.0, max(0.0, (self.OPEN_REFERENCE - self.ratio) / span)))

    @property
    def releasing(self) -> bool:
        """True while pinched but fingers are clearly separating (release likely)."""
        return self.active and self.ratio > (self.on + self.off) * 0.5

    def update(self, ratio: float) -> PinchTransition:
        self.ratio = ratio
        if not self.active:
            if ratio < self.on:
                self._count += 1
                if self._count >= self.on_frames:
                    self.active = True
                    self._count = 0
                    return PinchTransition.START
            else:
                self._count = 0
        else:
            if ratio > self.off:
                self._count += 1
                if self._count >= self.off_frames:
                    self.active = False
                    self._count = 0
                    return PinchTransition.END
            else:
                self._count = 0
        return PinchTransition.NONE
