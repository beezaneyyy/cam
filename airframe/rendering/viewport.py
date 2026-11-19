"""Mapping between camera pixel space and window (screen) space."""

from __future__ import annotations

import numpy as np


class Viewport:
    """Fits the camera image inside the window, preserving aspect ratio (letterbox)."""

    def __init__(self, camera_size: tuple[int, int] = (1280, 720), screen_size: tuple[int, int] = (1280, 720)):
        self.camera_size = (1, 1)
        self.screen_size = (1, 1)
        self.scale = 1.0
        self.offset = np.zeros(2)
        self.update(camera_size, screen_size)

    def update(self, camera_size: tuple[int, int], screen_size: tuple[int, int]) -> None:
        if camera_size == self.camera_size and screen_size == self.screen_size:
            return
        cw, ch = max(1, camera_size[0]), max(1, camera_size[1])
        sw, sh = max(1, screen_size[0]), max(1, screen_size[1])
        self.camera_size, self.screen_size = (cw, ch), (sw, sh)
        self.scale = min(sw / cw, sh / ch)
        self.offset = np.array([(sw - cw * self.scale) / 2.0, (sh - ch * self.scale) / 2.0])

    @property
    def size(self) -> tuple[int, int]:
        """Pixel size of the displayed camera image."""
        return (int(round(self.camera_size[0] * self.scale)), int(round(self.camera_size[1] * self.scale)))

    @property
    def rect(self) -> tuple[int, int, int, int]:
        w, h = self.size
        return (int(round(self.offset[0])), int(round(self.offset[1])), w, h)

    def to_screen(self, p) -> np.ndarray:
        return np.asarray(p, dtype=np.float64) * self.scale + self.offset

    def to_camera(self, p) -> np.ndarray:
        return (np.asarray(p, dtype=np.float64) - self.offset) / self.scale

    def length(self, v: float) -> float:
        return v * self.scale
