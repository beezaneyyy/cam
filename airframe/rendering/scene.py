"""Composition of the camera image with the frame effects (NumPy/OpenCV side)."""

from __future__ import annotations

import cv2
import numpy as np

from airframe.effects import EffectContext, EffectRegistry, apply_effect_region, polygon_bbox
from airframe.interaction.frame import VirtualFrame
from airframe.interaction.geometry import corners


class SceneComposer:
    """Builds the final camera image for display.

    Effects are applied in *camera* space (frames live there), then the result
    is resized once to the on-screen size. The camera frame is shared with the
    tracking thread, so it is copied exactly once (or resized, which also
    produces a new array) and never modified in place.
    """

    def __init__(self, registry: EffectRegistry, grade: float = 0.82):
        self.registry = registry
        self.grade = grade
        self._vignette: np.ndarray | None = None
        self._vignette_size: tuple[int, int] = (0, 0)

    def _vignette_for(self, w: int, h: int) -> np.ndarray:
        if self._vignette is None or self._vignette_size != (w, h):
            y, x = np.ogrid[-1:1:complex(0, h), -1:1:complex(0, w)]
            r = np.sqrt((x * 0.85) ** 2 + (y * 1.0) ** 2)
            v = np.clip(1.0 - 0.55 * np.clip(r - 0.35, 0, None) ** 1.6, 0.25, 1.0) * self.grade
            v8 = (v * 255).astype(np.uint8)
            self._vignette = cv2.merge([v8, v8, v8])
            self._vignette_size = (w, h)
        return self._vignette

    def compose(
        self,
        frame: np.ndarray,
        frames: list[VirtualFrame],
        effects_enabled: bool,
        t: float,
        out_size: tuple[int, int],
        post=None,
    ) -> np.ndarray:
        """``post(img)`` may draw onto the camera-space image before it is resized."""
        h, w = frame.shape[:2]
        # Darkened, vignetted base (also our one and only copy of the shared frame).
        img = cv2.multiply(frame, self._vignette_for(w, h), scale=1.0 / 255.0)
        if effects_enabled:
            for f in frames:
                if f.opacity <= 0.02 or f.effect not in self.registry:
                    continue
                poly = corners(f.display)
                ctx = EffectContext(time=t, seed=f.frame_id, intensity=f.opacity)
                # effects show the *ungraded* camera image so they read brighter than the background
                self._apply_with_source(img, frame, poly, f, ctx)
        if post is not None:
            post(img)
        if (w, h) != tuple(out_size):
            interp = cv2.INTER_AREA if out_size[0] < w else cv2.INTER_LINEAR
            img = cv2.resize(img, out_size, interpolation=interp)
        return img

    def _apply_with_source(self, img, source, poly, f: VirtualFrame, ctx: EffectContext) -> None:
        # Restore original brightness inside the region, then apply the effect on top.
        box = polygon_bbox(poly, img.shape[1], img.shape[0])
        if box is None:
            return
        x0, y0, x1, y1 = box
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(poly - (x0, y0)).astype(np.int32), 255)
        np.copyto(img[y0:y1, x0:x1], source[y0:y1, x0:x1], where=mask[..., None].astype(bool))
        apply_effect_region(img, poly, self.registry.get(f.effect), ctx, opacity=f.opacity)
