"""CUTOUT — cut a piece of reality and keep it as a movable sticker.

Uses the same gesture grammar as FRAMES (it reuses ``InteractionController``):

* Two pinches in empty space, pull apart, release → a short countdown (time to
  move your hands out of the way), then a flash: that region of the live image
  is frozen into a sticker, right where it was.
* Pinch a sticker: move it · two pinches on it: scale + rotate · corner: resize
* Drag it to the ⊗: delete · D: duplicate (clone yourself) · B: border style
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pygame

from airframe.config import InteractionConfig
from airframe.effects import EffectRegistry, polygon_bbox
from airframe.interaction import FrameManager, InteractionController, RectGeom
from airframe.interaction.controller import MOUSE_ID, Mode
from airframe.interaction.frame import VirtualFrame
from airframe.interaction.geometry import corners
from airframe.rendering import theme as T
from airframe.rendering.overlay import OverlayRenderer, corner_brackets, dashed_polygon
from airframe.rendering.viewport import Viewport

from .base import MOUSE_POINTER, Experience

CAPTURE_DELAY = 0.9          # seconds between release and the snapshot
BORDERS = ["sticker", "neon", "none"]


@dataclass
class Sticker:
    image: np.ndarray          # BGR, captured pixels
    captured_at: float


def warp_sticker(img: np.ndarray, sticker: np.ndarray, g: RectGeom, opacity: float = 1.0) -> np.ndarray | None:
    """Draw ``sticker`` onto ``img`` (in place) at geometry ``g``. Returns the polygon drawn, or None."""
    H, W = img.shape[:2]
    poly = corners(g)
    box = polygon_bbox(poly, W, H)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    sh, sw = sticker.shape[:2]
    # affine: sticker pixel → camera pixel, then shift into the ROI
    sx, sy = g.w / max(sw, 1), g.h / max(sh, 1)
    c, s = math.cos(g.angle), math.sin(g.angle)
    m = np.array(
        [[c * sx, -s * sy, 0.0], [s * sx, c * sy, 0.0]], np.float64
    )
    centre_src = np.array([sw / 2.0, sh / 2.0])
    m[:, 2] = np.array([g.cx - x0, g.cy - y0]) - m[:, :2] @ centre_src
    roi = img[y0:y1, x0:x1]
    warped = cv2.warpAffine(sticker, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, np.round(poly - (x0, y0)).astype(np.int32), 255, cv2.LINE_AA)
    a = (mask.astype(np.float32) * (opacity / 255.0))[..., None]
    roi[...] = (warped * a + roi * (1 - a)).astype(np.uint8)
    return poly


class CutoutMode(Experience):
    name, label, accent = "cutout", "Cutout", T.AMBER
    hints = [
        ("2 PINCHES + PULL", "cut"),
        ("PINCH", "move"),
        ("2 PINCHES ON IT", "scale · rotate"),
        ("D  B  DEL", "clone · border · delete"),
    ]

    def __init__(self, config: InteractionConfig | None = None) -> None:
        super().__init__()
        cfg = config or InteractionConfig()
        self.cfg = cfg
        self.manager = FrameManager(max_frames=20, min_size=cfg.min_frame_size)
        self.controller = InteractionController(self.manager, cfg, ["cutout"], "cutout", on_created=self._on_created)
        self.stickers: dict[int, Sticker] = {}
        self.pending: dict[int, float] = {}      # frame id → capture time
        self.flashes: list[tuple[RectGeom, float]] = []
        self.border = 0
        self.camera: np.ndarray | None = None
        self.now = 0.0
        self._gf = None
        self._chrome = OverlayRenderer(EffectRegistry())
        self._mouse_down = False
        self._hands = []

    # -- hooks ------------------------------------------------------------------------------
    def on_camera(self, image):
        self.camera = image

    def on_gestures(self, gf, now):
        self._gf = gf
        self._hands = gf.hands

    def exit(self):
        self.controller.cancel_all()
        self._mouse_down = False

    def _on_created(self, frame: VirtualFrame) -> None:
        self.pending[frame.frame_id] = self.now + CAPTURE_DELAY
        self.controller.notify("MOVE YOUR HANDS AWAY…")

    def capture(self, frame_id: int) -> bool:
        """Freeze the camera pixels under the frame into its sticker."""
        f = self.manager.get(frame_id)
        if f is None or self.camera is None:
            return False
        H, W = self.camera.shape[:2]
        g = f.target
        x0, y0 = max(0, int(round(g.cx - g.w / 2))), max(0, int(round(g.cy - g.h / 2)))
        x1, y1 = min(W, int(round(g.cx + g.w / 2))), min(H, int(round(g.cy + g.h / 2)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            self.manager.remove(frame_id, animate=False)
            return False
        self.stickers[frame_id] = Sticker(self.camera[y0:y1, x0:x1].copy(), self.now)
        # shrink the frame to what was actually captured (clipped at the image edge)
        f.target = RectGeom((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0)
        f.display = f.target
        self.flashes.append((f.target, self.now))
        self.controller.notify(f"CUT {len(self.stickers):02d}")
        return True

    def duplicate_selected(self) -> str | None:
        sel = self.manager.selected
        if sel is None or sel.frame_id not in self.stickers:
            return None
        g = sel.target
        clone = self.manager.create(RectGeom(g.cx + 40, g.cy + 30, g.w, g.h, g.angle), "cutout")
        if clone is None:
            return None
        self.stickers[clone.frame_id] = Sticker(self.stickers[sel.frame_id].image, self.now)
        self.manager.select(clone.frame_id)
        return "CLONED"

    # -- update -----------------------------------------------------------------------------
    def update(self, pointers, dt, now, size):
        super().update(pointers, dt, now, size)
        self.now = now
        for p in pointers:  # the mouse drives the same controller as a virtual hand
            if p.pid != MOUSE_POINTER:
                continue
            if p.started:
                self.controller.pointer_down(p.pos, now, MOUSE_ID)
                self._mouse_down = True
            elif p.pinching:
                self.controller.pointer_move(p.pos, MOUSE_ID)
            if p.ended and self._mouse_down:
                self.controller.pointer_up(p.pos, now, MOUSE_ID)
                self._mouse_down = False
        self.controller.update(self._gf, now)
        self._gf = None
        self.manager.update(dt, self.cfg.follow_speed)
        for fid, at in list(self.pending.items()):
            if now >= at:
                self.pending.pop(fid)
                self.capture(fid)
        alive = {f.frame_id for f in self.manager.frames}
        self.stickers = {k: v for k, v in self.stickers.items() if k in alive}
        self.pending = {k: v for k, v in self.pending.items() if k in alive}
        self.flashes = [(g, t) for g, t in self.flashes if now - t < 0.5]

    @property
    def notices(self):
        return self.controller.notices

    def handle_key(self, key, mod):
        if key == pygame.K_d:
            return self.duplicate_selected()
        if key == pygame.K_b:
            self.border = (self.border + 1) % len(BORDERS)
            return f"BORDER · {BORDERS[self.border].upper()}"
        if key in (pygame.K_DELETE, pygame.K_BACKSPACE, pygame.K_x):
            if mod & pygame.KMOD_SHIFT:
                return self.clear()
            return "STICKER REMOVED" if self.controller.delete_selected() else None
        return None

    def undo(self):
        live = [f for f in self.manager.active_frames if f.frame_id in self.stickers]
        if not live:
            return None
        self.manager.remove(live[-1].frame_id)
        return "UNDO"

    def clear(self):
        self.controller.cancel_all()
        self.manager.clear()
        self.pending.clear()
        return "ALL CUTS CLEARED"

    # -- rendering --------------------------------------------------------------------------
    def render_image(self, img, now):
        for f in self.manager.frames:
            st = self.stickers.get(f.frame_id)
            if st is None or f.opacity <= 0.02:
                continue
            g = f.display
            style = BORDERS[self.border]
            if style == "sticker":  # drop shadow + white die-cut border
                shadow = RectGeom(g.cx + 10, g.cy + 14, g.w + 14, g.h + 14, g.angle)
                poly = np.round(corners(shadow)).astype(np.int32)
                box = polygon_bbox(corners(shadow), img.shape[1], img.shape[0])
                if box is not None:
                    x0, y0, x1, y1 = box
                    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    cv2.fillConvexPoly(m, poly - (x0, y0), 255, cv2.LINE_AA)
                    m = cv2.GaussianBlur(m, (0, 0), 6)
                    roi = img[y0:y1, x0:x1]
                    roi[...] = (roi * (1 - m[..., None].astype(np.float32) / 255 * 0.55 * f.opacity)).astype(np.uint8)
                rim = RectGeom(g.cx, g.cy, g.w + 12, g.h + 12, g.angle)
                cv2.fillConvexPoly(img, np.round(corners(rim)).astype(np.int32), (245, 245, 245), cv2.LINE_AA)
            warp_sticker(img, st.image, g, f.opacity)

    def render_light(self, glow, now):
        if BORDERS[self.border] == "neon":
            for f in self.manager.active_frames:
                if f.frame_id in self.stickers:
                    glow.polyline(corners(f.display), (40, 190, 255), 3, closed=True)
        for g, t in self.flashes:  # camera-flash burst on capture
            k = 1 - (now - t) / 0.5
            col = tuple(int(255 * k) for _ in range(3))
            glow.circle((g.cx, g.cy), max(g.w, g.h) * 0.6, tuple(int(c * 0.6) for c in col), -1)
            glow.polyline(corners(RectGeom(g.cx, g.cy, g.w + 30 * (1 - k), g.h + 30 * (1 - k))), col, 4, closed=True)

    def render_overlay(self, layer, vp: Viewport, now):
        state = self.controller.state
        sel = self.manager.selected_id
        for f in self.manager.frames:
            pts = [vp.to_screen(p) for p in corners(f.display)]
            if f.frame_id in self.pending:  # countdown
                left = max(0.0, self.pending[f.frame_id] - now) / CAPTURE_DELAY
                dashed_polygon(layer, T.with_alpha(T.AMBER, 0.95), pts, 10, 6, 2, phase=-now * 60)
                c = vp.to_screen((f.display.cx, f.display.cy))
                rect = pygame.Rect(0, 0, 56, 56)
                rect.center = (c[0], c[1])
                pygame.draw.circle(layer, T.with_alpha(T.PANEL, 0.6), c, 30)
                pygame.draw.arc(layer, T.AMBER, rect, math.pi / 2, math.pi / 2 + left * math.tau, 3)
                txt = T.font(10, mono=True).render("HANDS AWAY", True, T.AMBER)
                layer.blit(txt, txt.get_rect(midtop=(c[0], c[1] + 34)))
            elif f.frame_id == sel or f.held:
                corner_brackets(layer, T.with_alpha(T.INK, 0.95), pts, 18, 2)
                for p in pts:
                    pygame.draw.circle(layer, T.with_alpha(T.AMBER, 0.95), p, 4, 1)
        if state.creation is not None:
            self._chrome._draw_creation(layer, vp, state.creation, state.creation_valid, now)
        if state.trash_active:
            self._chrome._draw_trash(layer, vp, state.trash_center, self.cfg.trash_radius, state.trash_hover, now)
        for h in self._hands:  # pinch cursors with the current action (MOVE / RESIZE / DRAW…)
            self._chrome._draw_cursor(layer, vp, h, state.modes.get(h.hand_id, Mode.IDLE), now)
        f = T.font(10, mono=True)
        s = f.render(f"{len(self.stickers):2d} CUTS  ·  BORDER {BORDERS[self.border].upper()}", True, T.INK_FAINT)
        layer.blit(s, (layer.get_width() - s.get_width() - 18, 130))
