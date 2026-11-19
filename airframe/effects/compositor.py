"""Apply an effect to a (possibly rotated) quadrilateral region of a frame, in place."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Effect, EffectContext


def polygon_bbox(polygon: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Integer bounding box of ``polygon`` clipped to the image. None if empty."""
    x0 = max(int(np.floor(polygon[:, 0].min())), 0)
    y0 = max(int(np.floor(polygon[:, 1].min())), 0)
    x1 = min(int(np.ceil(polygon[:, 0].max())), width)
    y1 = min(int(np.ceil(polygon[:, 1].max())), height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _is_axis_aligned(polygon: np.ndarray) -> bool:
    # TL→TR horizontal and TR→BR vertical (within half a pixel)
    return abs(polygon[0, 1] - polygon[1, 1]) < 0.5 and abs(polygon[1, 0] - polygon[2, 0]) < 0.5


def apply_effect_region(
    frame: np.ndarray,
    polygon: np.ndarray,
    effect: Effect,
    ctx: EffectContext,
    opacity: float = 1.0,
    decorate: bool = True,
) -> bool:
    """Run ``effect`` on the pixels inside ``polygon`` and write them back into ``frame``.

    Only the bounding box of the region is processed. For rotated regions a
    polygon mask restricts the write to the inside of the shape, so pixels
    outside the region are never altered (except by ``decorate``, e.g. glow).
    Returns False when the region is off-screen.
    """
    h, w = frame.shape[:2]
    polygon = np.asarray(polygon, dtype=np.float64)
    box = polygon_bbox(polygon, w, h)
    if box is None:
        return False
    x0, y0, x1, y1 = box
    roi = frame[y0:y1, x0:x1]
    out = effect.apply(roi, ctx)
    if out.shape != roi.shape or out.dtype != roi.dtype:
        raise ValueError(f"effect {effect.name!r} returned {out.shape}/{out.dtype}, expected {roi.shape}/{roi.dtype}")
    if opacity < 0.999:
        out = cv2.addWeighted(out, opacity, roi, 1.0 - opacity, 0)

    if _is_axis_aligned(polygon) and np.all(polygon >= 0) and np.all(polygon[:, 0] <= w) and np.all(polygon[:, 1] <= h):
        roi[...] = out
    else:
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        local = np.round(polygon - (x0, y0)).astype(np.int32)
        cv2.fillConvexPoly(mask, local, 255, lineType=cv2.LINE_8)
        np.copyto(roi, out, where=mask[..., None].astype(bool))

    if decorate:
        effect.decorate(frame, polygon, ctx)
    return True
