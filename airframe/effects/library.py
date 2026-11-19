"""The built-in effects.

All effects are vectorised OpenCV/NumPy operations sized for real time on a
modest CPU: they work on the region only and avoid per-pixel Python loops.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .base import Effect, EffectContext, EffectRegistry
from .extra import EXTRA_EFFECTS


def _offset_channel(channel: np.ndarray, delta: int) -> np.ndarray:
    """Saturating ``channel + delta``. (cv2.add with a Python scalar misbehaves on 1×1 inputs.)"""
    return np.clip(channel.astype(np.int16) + delta, 0, 255).astype(np.uint8)


class NeonEffect(Effect):
    """Glowing animated border; the inside gets a cool, slightly contrasted grade."""

    name, label, accent = "neon", "Neon Frame", (0, 240, 255)

    def apply(self, roi, ctx):
        # cool tint + contrast: B up, R down, slight darkening for the glow to pop
        out = cv2.convertScaleAbs(roi, alpha=1.08, beta=-12)
        out[..., 0] = _offset_channel(out[..., 0], 18)
        out[..., 2] = _offset_channel(out[..., 2], -10)
        return out

    def decorate(self, frame, polygon, ctx):
        h, w = frame.shape[:2]
        pulse = 0.75 + 0.25 * math.sin(ctx.time * 3.2 + ctx.seed)
        sigma = 4.0  # at half resolution
        pad = int(sigma * 3 * 2) + 8  # blur reach (3σ, upscaled ×2) so the glow never gets clipped
        x0 = max(int(polygon[:, 0].min()) - pad, 0)
        y0 = max(int(polygon[:, 1].min()) - pad, 0)
        x1 = min(int(polygon[:, 0].max()) + pad, w)
        y1 = min(int(polygon[:, 1].max()) + pad, h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        # Glow is rendered at half resolution (cheap blur) and added back.
        sw, sh = max((x1 - x0) // 2, 1), max((y1 - y0) // 2, 1)
        layer = np.zeros((sh, sw, 3), np.uint8)
        pts = ((polygon - (x0, y0)) / 2.0).astype(np.int32)
        color = (255, 220, 40)  # cyan in BGR
        magenta = (255, 60, 190)
        cv2.polylines(layer, [pts], True, color, 3, cv2.LINE_AA)
        glow = cv2.GaussianBlur(layer, (0, 0), sigma)
        glow = cv2.convertScaleAbs(glow, alpha=1.5 * pulse)
        glow = cv2.resize(glow, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        region = frame[y0:y1, x0:x1]
        region[...] = cv2.add(region, glow)  # saturating add, written back into the view
        # crisp core line + offset magenta ghost for a chromatic neon feel
        poly = polygon.astype(np.int32)
        cv2.polylines(frame, [poly + 2], True, magenta, 1, cv2.LINE_AA)
        cv2.polylines(frame, [poly], True, (255, 250, 200), 2, cv2.LINE_AA)


class RGBSplitEffect(Effect):
    name, label, accent = "rgb_split", "RGB Split", (255, 70, 120)

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        base = max(3, int(w * 0.026))
        off = int(base * (1.0 + 0.6 * math.sin(ctx.time * 2.7 + ctx.seed)))
        off = max(1, min(off, w // 4))
        voff = max(0, int(off * 0.35))
        out = roi.copy()
        # red channel → right, blue channel → left (+ a small vertical drift)
        out[:, off:, 2] = roi[:, :-off, 2]
        out[:, :off, 2] = roi[:, :1, 2]
        out[:, :-off, 0] = roi[:, off:, 0]
        out[:, -off:, 0] = roi[:, -1:, 0]
        if voff and h > voff:
            out[voff:, :, 0] = out[:-voff, :, 0].copy()
        return out


class PixelateEffect(Effect):
    name, label, accent = "pixelate", "Pixelation", (120, 255, 170)

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        block = max(6, int(min(w, h) / 22))
        sw, sh = max(1, w // block), max(1, h // block)
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        # quantise colours a little for a retro palette
        small = (small // 24) * 24 + 12
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class GlitchEffect(Effect):
    name, label, accent = "glitch", "Glitch", (200, 120, 255)

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi.copy()
        step = int(ctx.time * 14)  # new glitch pattern ~14 times per second
        rng = np.random.default_rng((step * 7919 + ctx.seed * 104729) & 0xFFFFFFFF)
        burst = rng.random() < 0.35  # occasional strong bursts
        out = roi.copy()
        n_bands = rng.integers(3, 9 if burst else 6)
        max_shift = max(2, int(w * (0.12 if burst else 0.04)))
        for _ in range(n_bands):
            y = int(rng.integers(0, h - 1))
            bh = int(rng.integers(2, max(3, h // (6 if burst else 14))))
            y1 = min(h, y + bh)
            shift = int(rng.integers(-max_shift, max_shift + 1))
            out[y:y1] = np.roll(roi[y:y1], shift, axis=1)
            if rng.random() < 0.5:  # channel tear inside the band
                ch = int(rng.integers(0, 3))
                out[y:y1, :, ch] = np.roll(roi[y:y1, :, ch], -shift // 2 + 3, axis=1)
        if burst and rng.random() < 0.5:
            # a solid colour block, like a corrupted macroblock
            bw, bh = int(rng.integers(w // 10 + 1, w // 3 + 2)), int(rng.integers(2, h // 8 + 3))
            x, y = int(rng.integers(0, max(1, w - bw))), int(rng.integers(0, max(1, h - bh)))
            out[y:y + bh, x:x + bw] = cv2.bitwise_not(out[y:y + bh, x:x + bw])
        # constant subtle chromatic offset
        out[:, 3:, 2] = out[:, :-3, 2]
        return out


class NegativeEffect(Effect):
    name, label, accent = "negative", "Negative", (240, 240, 240)

    def apply(self, roi, ctx):
        return cv2.bitwise_not(roi)


class ThermalEffect(Effect):
    name, label, accent = "thermal", "Thermal", (255, 140, 40)

    def apply(self, roi, ctx):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 2.0)
        # fixed contrast curve (per-frame min/max normalisation would flicker on video)
        gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=-20)
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


class ScanlinesEffect(Effect):
    name, label, accent = "scanlines", "Scanlines", (140, 255, 120)

    def __init__(self) -> None:
        self._cache: dict[int, np.ndarray] = {}

    def _pattern(self, h: int) -> np.ndarray:
        pat = self._cache.get(h)
        if pat is None:
            rows = np.arange(h)
            pat = np.where(rows % 3 == 2, 110, 256).astype(np.uint16)  # every 3rd row dark
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[h] = pat
        return pat

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        pat = self._pattern(h).copy()
        # rolling bright refresh bar
        bar_y = int((ctx.time * 0.35 + ctx.seed * 0.13) % 1.0 * (h + 60)) - 30
        y0, y1 = max(0, bar_y), min(h, bar_y + 30)
        if y1 > y0:
            pat[y0:y1] = np.minimum(pat[y0:y1] + 40, 300)
        flicker = 0.96 + 0.04 * math.sin(ctx.time * 50)
        scaled = (pat * flicker).astype(np.uint32)
        out = (roi.astype(np.uint32) * scaled[:, None, None]) >> 8  # uint32: 255*300 overflows uint16
        out = np.clip(out, 0, 255).astype(np.uint8)
        # phosphor tint
        out[..., 1] = _offset_channel(out[..., 1], 14)
        return out


class EdgeEffect(Effect):
    name, label, accent = "edges", "Edge Detection", (0, 200, 255)

    def apply(self, roi, ctx):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 130)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        out = (roi >> 3)  # very dark base keeps a hint of the scene
        glow = cv2.GaussianBlur(edges, (0, 0), 2.5)
        color = np.array([255, 230, 60], np.float32)  # cyan (BGR)
        add = (glow[..., None].astype(np.float32) * (color / 255.0) * 0.8).astype(np.uint8)
        out = cv2.add(out, add)
        out[edges > 0] = (255, 245, 170)
        return out


def default_registry() -> EffectRegistry:
    reg = EffectRegistry()
    for cls in (
        NeonEffect,
        RGBSplitEffect,
        PixelateEffect,
        GlitchEffect,
        NegativeEffect,
        ThermalEffect,
        ScanlinesEffect,
        EdgeEffect,
        *EXTRA_EFFECTS,
    ):
        reg.register(cls())
    return reg
