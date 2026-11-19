"""Vector overlay drawn with pygame on top of the camera image.

Everything here is in screen space; the ``Viewport`` maps camera coordinates.
Drawing happens on a transparent layer so lines can use alpha.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from airframe.effects import EffectRegistry
from airframe.gestures.engine import HandInfo
from airframe.interaction.controller import InteractionState, Mode
from airframe.interaction.frame import VirtualFrame
from airframe.interaction.geometry import RectGeom, corners
from airframe.tracking.landmarks import CONNECTIONS, INDEX_TIP, THUMB_TIP

from . import theme as T
from .viewport import Viewport

# colour *names*, resolved at draw time so the active theme applies
MODE_COLORS = {
    Mode.IDLE: "CYAN",
    Mode.ANCHOR: "CYAN",
    Mode.CREATE: "GREEN",
    Mode.DRAG: "BLUE",
    Mode.RESIZE: "AMBER",
    Mode.TRANSFORM: "VIOLET",
    Mode.CONSUMED: "INK_FAINT",
}

MODE_LABELS = {
    Mode.ANCHOR: "ANCHOR",
    Mode.CREATE: "DRAW",
    Mode.DRAG: "MOVE",
    Mode.RESIZE: "RESIZE",
    Mode.TRANSFORM: "SCALE · ROTATE",
}


def dashed_polygon(surf, color, pts, dash: float = 10, gap: float = 7, width: int = 1, phase: float = 0.0):
    n = len(pts)
    for i in range(n):
        a, b = np.asarray(pts[i], float), np.asarray(pts[(i + 1) % n], float)
        seg = b - a
        length = float(np.hypot(*seg))
        if length < 1:
            continue
        d = seg / length
        s = -(phase % (dash + gap))
        while s < length:
            s0, s1 = max(s, 0.0), min(s + dash, length)
            if s1 > s0:
                pygame.draw.line(surf, color, a + d * s0, a + d * s1, width)
            s += dash + gap


def corner_brackets(surf, color, pts, size: float, width: int = 2):
    """L-shaped brackets at each corner of a quadrilateral."""
    n = len(pts)
    for i in range(n):
        p = np.asarray(pts[i], float)
        for j in (i - 1, i + 1):
            q = np.asarray(pts[j % n], float)
            v = q - p
            length = float(np.hypot(*v))
            if length < 1:
                continue
            end = p + v / length * min(size, length * 0.4)
            pygame.draw.line(surf, color, p, end, width)


class OverlayRenderer:
    """Draws onto a transparent (SRCALPHA) layer owned by the caller."""

    def __init__(self, registry: EffectRegistry):
        self.registry = registry

    def draw(
        self,
        layer: pygame.Surface,
        vp: Viewport,
        frames: list[VirtualFrame],
        selected_id: int | None,
        hands: list[HandInfo],
        state: InteractionState,
        trash_radius: float,
        handle_radius: float,
        show_skeleton: bool,
        t: float,
    ) -> None:
        for f in frames:
            self._draw_frame(layer, vp, f, f.frame_id == selected_id, t)
        if state.zoom_frame is not None:
            self._draw_zoom(layer, vp, frames, hands, state, t)
        if state.creation is not None:
            self._draw_creation(layer, vp, state.creation, state.creation_valid, t)
        if state.trash_active:
            self._draw_trash(layer, vp, state.trash_center, trash_radius, state.trash_hover, t)
        for h in hands:
            mode = state.modes.get(h.hand_id, Mode.IDLE)
            if show_skeleton:
                self._draw_skeleton(layer, vp, h)
            self._draw_cursor(layer, vp, h, mode, t)
            self._draw_peace(layer, vp, h)

    # -- frames -------------------------------------------------------------------------
    def _draw_frame(self, surf, vp: Viewport, f: VirtualFrame, selected: bool, t: float):
        a = f.opacity
        if a <= 0.02:
            return
        eff = self.registry.get(f.effect) if f.effect in self.registry else None
        accent = eff.accent if eff else T.CYAN
        pts = [vp.to_screen(p) for p in corners(f.display)]
        base_alpha = (0.95 if selected or f.held else 0.55) * a
        pygame.draw.polygon(surf, T.with_alpha(accent, base_alpha * 0.55), pts, 1)
        size = max(10.0, min(26.0, min(f.display.w, f.display.h) * vp.scale * 0.18))
        corner_brackets(surf, T.with_alpha(T.INK if selected else accent, base_alpha), pts, size, 2 if selected else 1)

        if selected:
            for p in pts:  # resize handles
                r = 4 if not f.held else 5
                pygame.draw.circle(surf, T.with_alpha(T.BG, 0.85 * a), p, r + 2)
                pygame.draw.circle(surf, T.with_alpha(accent, a), p, r, 1)

        # label tag on the top edge
        label = f"{f.frame_id:02d}  {eff.label.upper() if eff else f.effect.upper()}"
        flash = max(0.0, 1.0 - (t - f.effect_changed_at) / 0.8)
        font = T.font(11, mono=True)
        txt = font.render(label, True, T.mix(T.INK, accent, 0.4 + 0.6 * flash))
        txt.set_alpha(int(255 * a * (0.95 if selected else 0.7)))
        tl, tr = pts[0], pts[1]
        ang = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
        if abs(ang) < 1.0:
            pad = 4
            bg = pygame.Rect(0, 0, txt.get_width() + pad * 2, txt.get_height() + 2)
            bg.bottomleft = (tl[0], tl[1] - 4)
            pygame.draw.rect(surf, T.with_alpha(T.BG, 0.55 * a), bg, border_radius=2)
            surf.blit(txt, (bg.x + pad, bg.y + 1))
        else:
            rot = pygame.transform.rotate(txt, -ang)
            normal = np.array([math.sin(math.radians(ang)), -math.cos(math.radians(ang))])
            anchor = (np.asarray(tl) + np.asarray(tr)) / 2 + normal * 12
            surf.blit(rot, rot.get_rect(center=(anchor[0], anchor[1])))

        # dimensions while being manipulated
        if f.held:
            g = f.display
            dims = f"{g.w:.0f} × {g.h:.0f}   {math.degrees(g.angle):+.0f}°"
            d = T.font(11, mono=True).render(dims, True, T.INK_DIM)
            c = vp.to_screen((g.cx, g.cy))
            bottom = max(p[1] for p in pts)
            surf.blit(d, d.get_rect(midtop=(c[0], bottom + 8)))

    def _draw_zoom(self, surf, vp: Viewport, frames, hands, state: InteractionState, t: float):
        frame = next((f for f in frames if f.frame_id == state.zoom_frame), None)
        hand = next((h for h in hands if h.hand_id == state.zoom_hand), None)
        if frame is not None:
            g = frame.display
            c = vp.to_screen((g.cx, g.cy))
            pts = [vp.to_screen(p) for p in corners(g)]
            dashed_polygon(surf, T.with_alpha(T.AMBER, 0.9), pts, 6, 6, 1, phase=-t * 30)
            label = f"ZOOM  {state.zoom_scale * 100:3.0f}%" if state.zoom_active else "ZOOM  ·  OPEN / CLOSE FINGERS"
            txt = T.font(13, mono=True, bold=True).render(label, True, T.AMBER)
            box = txt.get_rect(center=(c[0], c[1]))
            pygame.draw.rect(surf, T.with_alpha(T.BG, 0.7), box.inflate(18, 10), border_radius=4)
            surf.blit(txt, box)
            # ring that fills while the hand is still → size gets locked
            r = 14
            ring = pygame.Rect(0, 0, r * 2, r * 2)
            ring.midleft = (box.right + 18, box.centery)
            pygame.draw.circle(surf, T.with_alpha(T.INK_FAINT, 0.8), ring.center, r, 1)
            if state.zoom_lock > 0.02:
                pygame.draw.arc(surf, T.GREEN, ring, math.pi / 2, math.pi / 2 + state.zoom_lock * math.tau, 3)
        if hand is not None:
            a, b = vp.to_screen(hand.landmarks[THUMB_TIP, :2]), vp.to_screen(hand.landmarks[INDEX_TIP, :2])
            pygame.draw.line(surf, T.with_alpha(T.AMBER, 0.9), a, b, 2)
            for p in (a, b):
                pygame.draw.circle(surf, T.with_alpha(T.AMBER, 1.0), p, 6, 2)

    def _draw_creation(self, surf, vp: Viewport, g: RectGeom, valid: bool, t: float):
        color = T.GREEN if valid else T.RED
        pts = [vp.to_screen(p) for p in corners(g)]
        pygame.draw.polygon(surf, T.with_alpha(color, 0.08), pts)  # translucent fill on the alpha layer
        dashed_polygon(surf, T.with_alpha(color, 0.95), pts, 9, 6, 1, phase=-t * 40)
        corner_brackets(surf, T.with_alpha(color, 1.0), pts, 16, 2)
        txt = T.font(12, mono=True).render(f"{g.w:.0f} × {g.h:.0f}", True, color)
        surf.blit(txt, txt.get_rect(center=vp.to_screen((g.cx, g.cy))))

    def _draw_trash(self, surf, vp: Viewport, center, radius: float, hover: bool, t: float):
        c = vp.to_screen(center)
        r = vp.length(radius)
        color = T.RED if hover else T.INK_DIM
        if hover:
            pulse = 0.5 + 0.5 * math.sin(t * 10)
            pygame.draw.circle(surf, T.with_alpha(T.RED, 0.12 + 0.1 * pulse), c, r)
        pygame.draw.circle(surf, T.with_alpha(color, 0.8), c, r, 1)
        dashed_circle_r = r + 6
        for i in range(24):
            a0 = i / 24 * math.tau + t * 0.6
            p0 = c + dashed_circle_r * np.array([math.cos(a0), math.sin(a0)])
            p1 = c + dashed_circle_r * np.array([math.cos(a0 + 0.12), math.sin(a0 + 0.12)])
            pygame.draw.line(surf, T.with_alpha(color, 0.5), p0, p1, 1)
        s = r * 0.22
        pygame.draw.line(surf, T.with_alpha(color, 0.95), c - (s, s), c + (s, s), 2)
        pygame.draw.line(surf, T.with_alpha(color, 0.95), c + (-s, s), c + (s, -s), 2)
        lbl = T.font(10, mono=True).render("RELEASE TO DELETE" if hover else "DROP TO DELETE", True, color)
        surf.blit(lbl, lbl.get_rect(midbottom=(c[0], c[1] - r - 12)))

    def draw_hands(self, layer, vp: Viewport, hands: list[HandInfo], show_skeleton: bool, t: float) -> None:
        """Skeleton + mode-switch progress, for modes that draw their own cursors."""
        for h in hands:
            if show_skeleton:
                self._draw_skeleton(layer, vp, h)
            self._draw_peace(layer, vp, h)

    def _draw_peace(self, surf, vp: Viewport, h: HandInfo) -> None:
        if h.peace_progress <= 0.02:
            return
        c = vp.to_screen(h.palm_center)
        r = 34
        rect = pygame.Rect(0, 0, r * 2, r * 2)
        rect.center = (c[0], c[1])
        pygame.draw.circle(surf, T.with_alpha(T.INK, 0.25), c, r, 1)
        pygame.draw.arc(surf, T.with_alpha(T.AMBER, 1.0), rect, math.pi / 2, math.pi / 2 + h.peace_progress * math.tau, 3)
        txt = T.font(10, mono=True).render("NEXT MODE", True, T.AMBER)
        surf.blit(txt, txt.get_rect(midtop=(c[0], c[1] + r + 4)))

    # -- hands --------------------------------------------------------------------------
    def _draw_skeleton(self, surf, vp: Viewport, h: HandInfo):
        color = T.HAND_COLORS.get(h.handedness, T.BLUE)
        alpha = 0.25 if h.stale else 0.55
        pts = [vp.to_screen(p[:2]) for p in h.landmarks]
        for a, b in CONNECTIONS:
            pygame.draw.aaline(surf, T.with_alpha(color, alpha), pts[a], pts[b])
        for i, p in enumerate(pts):
            r = 3 if i in (THUMB_TIP, INDEX_TIP) else 2
            pygame.draw.circle(surf, T.with_alpha(color, alpha + 0.2), p, r)

    def _draw_cursor(self, surf, vp: Viewport, h: HandInfo, mode: Mode, t: float):
        color = getattr(T, MODE_COLORS.get(mode, "CYAN")) if h.pinching else T.HAND_COLORS.get(h.handedness, T.CYAN)
        c = vp.to_screen(h.interaction_point)
        if h.pinching:
            pygame.draw.circle(surf, T.with_alpha(color, 0.25), c, 16)
            pygame.draw.circle(surf, T.with_alpha(color, 1.0), c, 16, 2)
            pygame.draw.circle(surf, T.with_alpha(T.INK, 1.0), c, 3)
            label = MODE_LABELS.get(mode)
            if label:
                txt = T.font(10, mono=True).render(label, True, color)
                surf.blit(txt, (c[0] + 22, c[1] - txt.get_height() / 2))
        else:
            # ring contracts as thumb and index approach: live pinch feedback
            p = h.pinch_progress
            r = 30 - 16 * p
            pygame.draw.circle(surf, T.with_alpha(color, 0.25 + 0.6 * p), c, r, 1)
            if p > 0.05:
                sweep = p * math.tau
                rect = pygame.Rect(0, 0, (r + 5) * 2, (r + 5) * 2)
                rect.center = (c[0], c[1])
                pygame.draw.arc(surf, T.with_alpha(color, 0.9), rect, math.pi / 2, math.pi / 2 + sweep, 2)
            pygame.draw.circle(surf, T.with_alpha(color, 0.8), c, 2)
            if h.is_open:
                txt = T.font(9, mono=True).render("OPEN", True, T.INK_FAINT)
                surf.blit(txt, txt.get_rect(midtop=(c[0], c[1] + r + 4)))
