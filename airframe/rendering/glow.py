"""Additive light layer for neon strokes, particles and lightning.

Shapes are drawn (in camera coordinates) onto a black half-resolution canvas.
``composite`` blurs that canvas for the halo, upscales it and adds both the
halo and the sharp core onto the image with saturating addition — the classic
cheap bloom.
"""

from __future__ import annotations

import cv2
import numpy as np


class GlowLayer:
    def __init__(self, scale: float = 0.5):
        self.scale = scale
        self.canvas: np.ndarray | None = None
        self.dirty = False

    def begin(self, size: tuple[int, int]) -> None:
        """``size`` = (width, height) of the target image."""
        w, h = max(1, int(size[0] * self.scale)), max(1, int(size[1] * self.scale))
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), np.uint8)
        else:
            self.canvas[...] = 0
        self.dirty = False

    def _p(self, p) -> tuple[int, int]:
        return int(round(p[0] * self.scale)), int(round(p[1] * self.scale))

    def line(self, p0, p1, color, width: float = 2.0) -> None:
        cv2.line(self.canvas, self._p(p0), self._p(p1), color, max(1, int(round(width * self.scale))), cv2.LINE_AA)
        self.dirty = True

    def polyline(self, pts, color, width: float = 2.0, closed: bool = False) -> None:
        if len(pts) < 2:
            return
        arr = np.round(np.asarray(pts, np.float64) * self.scale).astype(np.int32)
        cv2.polylines(self.canvas, [arr], closed, color, max(1, int(round(width * self.scale))), cv2.LINE_AA)
        self.dirty = True

    def circle(self, c, r: float, color, thickness: float = -1) -> None:
        t = -1 if thickness < 0 else max(1, int(round(thickness * self.scale)))
        cv2.circle(self.canvas, self._p(c), max(1, int(round(r * self.scale))), color, t, cv2.LINE_AA)
        self.dirty = True

    def points(self, pts: np.ndarray, colors: np.ndarray) -> None:
        """Vectorised splat of many points (N,2) with colours (N,3) uint8 — 2×2 px each."""
        if len(pts) == 0:
            return
        h, w = self.canvas.shape[:2]
        xy = np.round(pts * self.scale).astype(np.intp)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < h - 1)
        xy, colors = xy[ok], colors[ok]
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            self.canvas[xy[:, 1] + dy, xy[:, 0] + dx] = np.maximum(self.canvas[xy[:, 1] + dy, xy[:, 0] + dx], colors)
        self.dirty = True

    def composite(self, img: np.ndarray, halo: float = 1.6, core: float = 1.0, sigma: float = 5.0) -> None:
        """Add the layer onto ``img`` (same size as given to ``begin``), in place."""
        if not self.dirty or self.canvas is None:
            return
        h, w = img.shape[:2]
        blurred = cv2.GaussianBlur(self.canvas, (0, 0), sigma)
        light = cv2.addWeighted(blurred, halo, self.canvas, core, 0)
        light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.add(img, light, dst=img)
