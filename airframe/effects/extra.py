"""Additional effects (second pack).

Same contract as ``library.py``: take a BGR uint8 ROI (read-only view),
return a new array of identical shape. Everything is vectorised; glyph/dot
patterns are pre-rendered into atlases and composed with fancy indexing.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .base import Effect, EffectContext


def _cell_grid(roi: np.ndarray, cw: int, ch: int):
    """Downsample to a grid of cells. Returns (cells BGR, rows, cols) or None if too small."""
    h, w = roi.shape[:2]
    cols, rows = w // cw, h // ch
    if cols < 1 or rows < 1:
        return None
    cells = cv2.resize(roi, (cols, rows), interpolation=cv2.INTER_AREA)
    return cells, rows, cols


def _place(roi: np.ndarray, tiles: np.ndarray, fill: np.ndarray | None = None) -> np.ndarray:
    """Put a (rows*ch, cols*cw, 3) tiled image into a ROI-sized canvas (remainder = fill)."""
    out = np.empty_like(roi) if fill is None else fill
    if fill is None:
        out[...] = 0
    th, tw = tiles.shape[:2]
    out[:th, :tw] = tiles
    return out


class KaleidoscopeEffect(Effect):
    name, label, accent = "kaleidoscope", "Kaleidoscope", (255, 120, 220)

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi.copy()
        hh, hw = h // 2, w // 2
        # slowly drifting source window → the pattern keeps evolving
        dx = int((math.sin(ctx.time * 0.6 + ctx.seed) * 0.5 + 0.5) * (w - hw))
        dy = int((math.cos(ctx.time * 0.45 + ctx.seed) * 0.5 + 0.5) * (h - hh))
        q = roi[dy:dy + hh, dx:dx + hw]
        top = np.hstack([q, q[:, ::-1]])
        full = np.vstack([top, top[::-1]])
        out = roi.copy()
        out[: full.shape[0], : full.shape[1]] = full
        return out


class AsciiEffect(Effect):
    name, label, accent = "ascii", "ASCII", (90, 255, 140)
    CHARS = " .:-=+*#%@"
    CW, CH = 8, 12

    def __init__(self) -> None:
        n = len(self.CHARS)
        atlas = np.zeros((n, self.CH, self.CW), np.uint8)
        for i, c in enumerate(self.CHARS):
            cv2.putText(atlas[i], c, (0, self.CH - 3), cv2.FONT_HERSHEY_PLAIN, 0.75, 255, 1, cv2.LINE_AA)
        self.atlas = atlas

    def apply(self, roi, ctx):
        grid = _cell_grid(roi, self.CW, self.CH)
        if grid is None:
            return (roi >> 2).copy()
        cells, rows, cols = grid
        lum = cv2.cvtColor(cells, cv2.COLOR_BGR2GRAY)
        idx = (lum.astype(np.uint16) * (len(self.CHARS) - 1) // 255).astype(np.intp)
        glyphs = self.atlas[idx]                                   # rows, cols, CH, CW
        mask = glyphs.transpose(0, 2, 1, 3).reshape(rows * self.CH, cols * self.CW)
        # colour each glyph with its cell colour, boosted for a terminal glow
        color = cv2.resize(cells, (cols * self.CW, rows * self.CH), interpolation=cv2.INTER_NEAREST)
        color = cv2.convertScaleAbs(color, alpha=1.4, beta=30)
        tiles = (color.astype(np.uint16) * mask[..., None] // 255).astype(np.uint8)
        return _place(roi, tiles)


class HalftoneEffect(Effect):
    name, label, accent = "halftone", "Halftone", (255, 220, 90)
    CELL = 10
    LEVELS = 16

    def __init__(self) -> None:
        c = self.CELL
        yy, xx = np.mgrid[0:c, 0:c]
        r = np.hypot(yy - (c - 1) / 2, xx - (c - 1) / 2)
        self.dots = np.stack(
            [np.clip((c * 0.72 * i / (self.LEVELS - 1)) - r + 0.5, 0, 1) for i in range(self.LEVELS)]
        ).astype(np.float32)                                       # LEVELS, c, c — antialiased discs

    def apply(self, roi, ctx):
        c = self.CELL
        grid = _cell_grid(roi, c, c)
        if grid is None:
            return roi.copy()
        cells, rows, cols = grid
        lum = cv2.cvtColor(cells, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        idx = np.clip((lum * (self.LEVELS - 1)).round(), 0, self.LEVELS - 1).astype(np.intp)
        mask = self.dots[idx].transpose(0, 2, 1, 3).reshape(rows * c, cols * c)
        paper = np.array([215, 235, 245], np.float32)              # warm paper (BGR)
        ink = cv2.resize(cells, (cols * c, rows * c), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        ink = ink * 1.15 + 10
        tiles = paper * (1 - mask[..., None]) * 0.2 + ink * mask[..., None]
        out = np.full_like(roi, 30)
        return _place(roi, np.clip(tiles, 0, 255).astype(np.uint8), out)


class PopArtEffect(Effect):
    name, label, accent = "pop_art", "Pop Art", (255, 230, 0)
    PALETTES = [
        [(40, 10, 60), (200, 40, 230), (40, 220, 255), (240, 250, 255)],    # violet / pink / yellow
        [(60, 20, 10), (230, 120, 30), (60, 230, 120), (220, 255, 250)],    # blue / green
        [(20, 20, 20), (60, 60, 230), (40, 200, 255), (250, 250, 250)],     # red / yellow
    ]

    def apply(self, roi, ctx):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        pal = np.array(self.PALETTES[int(ctx.time / 1.6 + ctx.seed) % len(self.PALETTES)], np.uint8)
        levels = np.minimum(gray.astype(np.uint16) * len(pal) // 256, len(pal) - 1)
        out = pal[levels]
        edges = cv2.Canny(gray, 60, 150)
        out[edges > 0] = (10, 10, 10)
        return out


class FrostedEffect(Effect):
    name, label, accent = "frosted", "Frosted Glass", (200, 230, 255)

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        sw, sh = max(1, w // 8), max(1, h // 8)
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 1.6)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        out = cv2.addWeighted(out, 0.82, np.full_like(out, 255), 0.18, 0)   # milky glass
        # grain that stays put (seeded per frame id) so it looks like glass, not noise
        rng = np.random.default_rng(ctx.seed)
        grain = rng.integers(-10, 11, (h, w, 1), dtype=np.int16)
        return np.clip(out.astype(np.int16) + grain, 0, 255).astype(np.uint8)


class NightVisionEffect(Effect):
    name, label, accent = "night_vision", "Night Vision", (80, 255, 80)

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=25)        # light amplification
        rng = np.random.default_rng(int(ctx.time * 30) + ctx.seed * 977)
        noise = rng.integers(-28, 29, (h, w), dtype=np.int16)
        g = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # circular scope vignette
        yy, xx = np.ogrid[0:h, 0:w]
        r = np.hypot((xx - w / 2) / max(w / 2, 1), (yy - h / 2) / max(h / 2, 1))
        vig = np.clip(1.25 - r * 0.9, 0.0, 1.0)
        g = (g * vig).astype(np.uint8)
        out = np.zeros_like(roi)
        out[..., 1] = g
        out[..., 0] = g >> 3
        out[..., 2] = g >> 2
        return out


class SketchEffect(Effect):
    name, label, accent = "sketch", "Pencil Sketch", (230, 230, 230)

    def apply(self, roi, ctx):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(255 - gray, (0, 0), 6)
        sketch = cv2.divide(gray, 255 - blur, scale=256)             # colour dodge
        sketch = cv2.convertScaleAbs(sketch, alpha=1.0, beta=-15)
        out = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        # faint paper tint
        out[..., 0] = (out[..., 0].astype(np.uint16) * 235 // 255).astype(np.uint8)
        return out


class HologramEffect(Effect):
    name, label, accent = "hologram", "Hologram", (60, 200, 255)

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        base = cv2.convertScaleAbs(gray, alpha=0.75, beta=10)
        out = np.zeros_like(roi)
        out[..., 0] = cv2.add(base, edges)                          # blue
        out[..., 1] = cv2.add((base.astype(np.uint16) * 200 // 255).astype(np.uint8), edges)
        out[..., 2] = base >> 3
        rows = np.arange(h)
        lines = ((rows + int(ctx.time * 40)) % 4 < 2).astype(np.uint8)
        out = out * (0.55 + 0.45 * lines)[:, None, None]
        flicker = 0.85 + 0.15 * math.sin(ctx.time * 23 + ctx.seed) * math.sin(ctx.time * 7)
        out = (out * flicker).astype(np.uint8)
        # occasional horizontal jitter of a band
        if int(ctx.time * 6 + ctx.seed) % 5 == 0 and h > 8:
            y = int((ctx.time * 97) % (h - 4))
            out[y:y + 4] = np.roll(out[y:y + 4], 6, axis=1)
        return out


class RippleEffect(Effect):
    name, label, accent = "ripple", "Liquid Ripple", (80, 180, 255)

    def __init__(self) -> None:
        self._grid: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi.copy()
        grid = self._grid.get((h, w))
        if grid is None:
            if len(self._grid) > 16:
                self._grid.clear()
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            grid = self._grid[(h, w)] = (xx, yy)
        xx, yy = grid
        cx, cy = w / 2, h / 2
        r = np.hypot(xx - cx, yy - cy)
        amp = max(2.0, min(w, h) * 0.018)
        wave = np.sin(r / max(min(w, h) * 0.045, 2) - ctx.time * 6.0).astype(np.float32) * amp
        nx = (xx - cx) / (r + 1e-3)
        ny = (yy - cy) / (r + 1e-3)
        return cv2.remap(roi, xx + nx * wave, yy + ny * wave, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class VHSEffect(Effect):
    name, label, accent = "vhs", "VHS", (255, 100, 160)

    def apply(self, roi, ctx):
        h, w = roi.shape[:2]
        ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        k = max(3, (w // 40) | 1)
        ycc[..., 1] = cv2.blur(ycc[..., 1], (k, 1))                 # chroma bleed
        ycc[..., 2] = cv2.blur(ycc[..., 2], (k, 1))
        shift = max(1, w // 120)
        ycc[:, shift:, 1] = ycc[:, :-shift, 1].copy()
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        out = cv2.addWeighted(out, 0.88, np.full_like(out, 40), 0.12, 0)  # washed-out tape
        rng = np.random.default_rng(int(ctx.time * 25) + ctx.seed * 31)
        # tracking band rolling down the frame
        band_y = int((ctx.time * 0.25 + ctx.seed * 0.17) % 1.0 * h)
        bh = max(2, h // 18)
        y0, y1 = band_y, min(h, band_y + bh)
        if y1 > y0:
            out[y0:y1] = np.roll(out[y0:y1], int(rng.integers(4, 14)), axis=1)
            noise = rng.integers(0, 90, (y1 - y0, w, 1), dtype=np.uint8)
            out[y0:y1] = cv2.add(out[y0:y1], np.repeat(noise, 3, axis=2))
        # sparse white dropout lines
        for _ in range(int(rng.integers(0, 3))):
            y = int(rng.integers(0, h))
            x = int(rng.integers(0, max(1, w - 10)))
            out[y, x:x + int(rng.integers(5, max(6, w // 4)))] = 230
        return out


EXTRA_EFFECTS = (
    KaleidoscopeEffect,
    AsciiEffect,
    HalftoneEffect,
    PopArtEffect,
    FrostedEffect,
    NightVisionEffect,
    SketchEffect,
    HologramEffect,
    RippleEffect,
    VHSEffect,
)
