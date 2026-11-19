"""Heads-up display: brand, camera status, telemetry, hand indicators, effect bar, toasts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pygame

from airframe.camera import CameraStatus
from airframe.effects import EffectRegistry
from airframe.interaction.controller import Mode, Notice
from airframe.rendering import theme as T

from .widgets import Button


@dataclass
class HandBadge:
    handedness: str
    pinching: bool
    mode: Mode
    stale: bool = False


@dataclass
class HudData:
    camera_status: CameraStatus
    camera_resolution: tuple[int, int]
    camera_fps: float
    render_fps: float
    tracking_fps: float
    inference_ms: float
    hands: list[HandBadge]
    current_effect: str
    effect_target: str          # "FRAME 03" or "NEW FRAMES"
    effects_enabled: bool
    frame_count: int
    notices: list[Notice] = field(default_factory=list)
    now: float = 0.0
    modes: list[tuple[str, str, tuple[int, int, int]]] = field(default_factory=list)  # (name, label, accent)
    mode_index: int = 0
    hints: list[tuple[str, str]] = field(default_factory=list)


STATUS_STYLE = {  # colour names, resolved at draw time (theme-aware)
    CameraStatus.RUNNING: ("GREEN", "LIVE"),
    CameraStatus.STARTING: ("AMBER", "STARTING"),
    CameraStatus.RECONNECTING: ("AMBER", "RECONNECTING"),
    CameraStatus.ERROR: ("RED", "ERROR"),
    CameraStatus.STOPPED: ("INK_FAINT", "OFF"),
}


def retro_panel(surf, rect: pygame.Rect, radius: int = 8) -> None:
    """Handheld-menu panel: light fill, blue outline, hard drop shadow. No-op in the neon theme."""
    if not T.is_retro():
        return
    shadow = rect.move(0, 3)
    pygame.draw.rect(surf, T.with_alpha(T.INK, 0.25), shadow, border_radius=radius)
    pygame.draw.rect(surf, T.with_alpha(T.PANEL, 0.92), rect, border_radius=radius)
    pygame.draw.rect(surf, T.BLUE, rect, 2, border_radius=radius)


def spaced(text: str, size: int, color, spacing: int = 3, mono: bool = False, bold: bool = False) -> pygame.Surface:
    """Render text with letter spacing (tracking) for headings."""
    f = T.font(size, mono=mono, bold=bold)
    glyphs = [f.render(ch, True, color) for ch in text]
    w = sum(g.get_width() for g in glyphs) + spacing * max(0, len(glyphs) - 1)
    surf = pygame.Surface((max(1, w), f.get_height()), pygame.SRCALPHA)
    x = 0
    for g in glyphs:
        surf.blit(g, (x, 0))
        x += g.get_width() + spacing
    return surf


class Hud:
    MARGIN = 18

    def __init__(
        self,
        registry: EffectRegistry,
        on_select_effect: Callable[[str], None],
        buttons: list[Button],
        on_select_mode: Callable[[int], None] = lambda i: None,
    ):
        self.registry = registry
        self.on_select_effect = on_select_effect
        self.on_select_mode = on_select_mode
        self.buttons = buttons
        self._chip_rects: list[tuple[pygame.Rect, str]] = []
        self._tab_rects: list[tuple[pygame.Rect, int]] = []
        self._hover_chip: str | None = None

    # -- input ----------------------------------------------------------------------------
    def handle(self, event: pygame.event.Event) -> bool:
        for b in self.buttons:
            if b.handle(event):
                return True
        if event.type == pygame.MOUSEMOTION:
            self._hover_chip = next((n for r, n in self._chip_rects if r.collidepoint(event.pos)), None)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, idx in self._tab_rects:
                if rect.collidepoint(event.pos):
                    self.on_select_mode(idx)
                    return True
            for rect, name in self._chip_rects:
                if rect.collidepoint(event.pos):
                    self.on_select_effect(name)
                    return True
        return False

    def hit(self, pos) -> bool:
        return (
            any(b.rect.collidepoint(pos) for b in self.buttons)
            or any(r.collidepoint(pos) for r, _ in self._chip_rects)
            or any(r.collidepoint(pos) for r, _ in self._tab_rects)
        )

    # -- drawing --------------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, data: HudData, view_rect: pygame.Rect) -> None:
        self._draw_view_marks(surf, view_rect)
        self._draw_brand(surf, data)
        self._draw_buttons(surf)
        self._draw_telemetry(surf, data)
        self._draw_mode_tabs(surf, data)
        if not data.modes or data.modes[data.mode_index][0] == "frames":
            self._draw_effect_bar(surf, data)
        else:
            self._chip_rects = []
            self._draw_hints(surf, data)
        self._draw_notices(surf, data)

    def _draw_mode_tabs(self, surf, data: HudData):
        self._tab_rects = []
        if not data.modes:
            return
        W = surf.get_width()
        f = T.font(11, mono=True)
        widths = [f.size(label.upper())[0] + 26 for _, label, _ in data.modes]
        total = sum(widths) + 2 * (len(widths) - 1)
        x = (W - total) // 2
        y = self.MARGIN
        bg = pygame.Rect(x - 4, y - 4, total + 8, 32)
        pygame.draw.rect(surf, T.with_alpha(T.PANEL, 0.6), bg, border_radius=6)
        retro_panel(surf, bg, radius=8)
        for i, ((name, label, accent), w) in enumerate(zip(data.modes, widths)):
            rect = pygame.Rect(x, y, w, 24)
            self._tab_rects.append((rect, i))
            active = i == data.mode_index
            if active and T.is_retro():  # solid "selected" pill, light text
                pygame.draw.rect(surf, T.BLUE, rect, border_radius=6)
            elif active:
                pygame.draw.rect(surf, T.with_alpha(accent, 0.18), rect, border_radius=4)
                pygame.draw.line(surf, accent, (rect.x + 8, rect.bottom - 1), (rect.right - 8, rect.bottom - 1), 2)
            color = (T.PANEL if T.is_retro() else T.INK) if active else (T.INK_DIM if T.is_retro() else T.INK_FAINT)
            txt = f.render(label.upper(), True, color)
            surf.blit(txt, txt.get_rect(center=rect.center))
            x += w + 2
        hint = T.font(9, mono=True).render("TAB  ·  HOLD V-SIGN TO SWITCH", True, T.INK_DIM if T.is_retro() else T.INK_FAINT)
        hr = hint.get_rect(midtop=(W // 2, bg.bottom + 6))
        retro_panel(surf, hr.inflate(16, 6), radius=4)
        surf.blit(hint, hr)

    def _draw_hints(self, surf, data: HudData):
        W, H = surf.get_size()
        if not data.hints:
            return
        fk, fv = T.font(11, mono=True, bold=True), T.font(11, mono=True)
        items = [(fk.render(k, True, T.INK), fv.render(v, True, T.INK_DIM)) for k, v in data.hints]
        widths = [a.get_width() + b.get_width() + 10 for a, b in items]
        total = sum(widths) + 28 * (len(items) - 1)
        x = (W - total) // 2
        y = H - self.MARGIN - 30
        bg = pygame.Rect(x - 16, y, total + 32, 30)
        pygame.draw.rect(surf, T.with_alpha(T.PANEL, 0.7), bg, border_radius=4)
        accent = data.modes[data.mode_index][2] if data.modes else T.CYAN
        pygame.draw.line(surf, accent, (bg.x + 10, bg.bottom - 1), (bg.x + 60, bg.bottom - 1), 2)
        for (ks, vs), w in zip(items, widths):
            surf.blit(ks, (x, y + 8))
            surf.blit(vs, (x + ks.get_width() + 10, y + 8))
            x += w + 28

    def _draw_view_marks(self, surf, r: pygame.Rect):
        c = T.with_alpha(T.INK, 0.35)
        s = 22
        for (x, y), (dx, dy) in (
            ((r.left + 8, r.top + 8), (1, 1)),
            ((r.right - 9, r.top + 8), (-1, 1)),
            ((r.left + 8, r.bottom - 9), (1, -1)),
            ((r.right - 9, r.bottom - 9), (-1, -1)),
        ):
            pygame.draw.line(surf, c, (x, y), (x + s * dx, y), 1)
            pygame.draw.line(surf, c, (x, y), (x, y + s * dy), 1)
        # centre reticle
        cx, cy = r.center
        pygame.draw.line(surf, T.with_alpha(T.INK, 0.12), (cx - 6, cy), (cx + 6, cy))
        pygame.draw.line(surf, T.with_alpha(T.INK, 0.12), (cx, cy - 6), (cx, cy + 6))

    def _draw_brand(self, surf, data: HudData):
        m = self.MARGIN
        retro_panel(surf, pygame.Rect(m - 8, m - 8, 270, 50))
        title = spaced("AIRFRAME", 15, T.INK, 5, bold=True)
        surf.blit(title, (m + 6, m))
        pygame.draw.line(surf, T.CYAN, (m, m + 3), (m, m + title.get_height() - 3), 2)
        cname, label = STATUS_STYLE.get(data.camera_status, ("INK_FAINT", "?"))
        color = getattr(T, cname)
        y = m + title.get_height() + 6
        pulse = 0.6 + 0.4 * math.sin(data.now * 4) if data.camera_status is not CameraStatus.RUNNING else 1.0
        pygame.draw.circle(surf, T.with_alpha(color, pulse), (m + 4, y + 7), 3)
        w, h = data.camera_resolution
        info = f"CAM {label}  ·  {w}×{h}  ·  {data.camera_fps:4.1f} FPS"
        surf.blit(T.font(10, mono=True).render(info, True, T.INK_DIM), (m + 14, y))

    def _draw_buttons(self, surf):
        m = self.MARGIN
        y = m + 52
        for b in self.buttons:
            b.rect = pygame.Rect(m, y, 118, 24)
            b.draw(surf)
            y += 30

    def _draw_telemetry(self, surf, data: HudData):
        m = self.MARGIN
        W = surf.get_width()
        f = T.font(10, mono=True)
        retro_panel(surf, pygame.Rect(W - m - 160, m - 8, 168, 68))
        lines = [
            ("RENDER", f"{data.render_fps:5.1f}"),
            ("TRACK", f"{data.tracking_fps:5.1f}"),
            ("INFER", f"{data.inference_ms:4.1f}ms"),
            ("FRAMES", f"{data.frame_count:5d}"),
        ]
        y = m
        for k, v in lines:
            vs = f.render(v, True, T.INK)
            ks = f.render(k, True, T.INK_FAINT)
            surf.blit(vs, (W - m - vs.get_width(), y))
            surf.blit(ks, (W - m - 74 - ks.get_width() + 20, y))
            y += 14
        # hand badges
        y += 8
        slots = {"Left": None, "Right": None}
        extra = []
        for h in data.hands:
            if h.handedness in slots and slots[h.handedness] is None:
                slots[h.handedness] = h
            else:
                extra.append(h)
        for name in ("Left", "Right"):  # fill the free slot with an unlabeled extra hand
            if slots[name] is None and extra:
                slots[name] = extra.pop()
        x = W - m
        for name in ("Right", "Left"):
            h = slots[name]
            color = T.HAND_COLORS[name]
            rect = pygame.Rect(0, 0, 58, 22)
            rect.topright = (x, y)
            if h is None:
                pygame.draw.rect(surf, T.with_alpha(T.INK_FAINT, 0.9), rect, 1, border_radius=11)
                txt = f.render(name[0], True, T.INK_FAINT)
                surf.blit(txt, txt.get_rect(center=rect.center))
            else:
                fill = 0.35 if h.pinching else 0.12
                pygame.draw.rect(surf, T.with_alpha(color, fill), rect, border_radius=11)
                pygame.draw.rect(surf, T.with_alpha(color, 0.5 if h.stale else 1.0), rect, 1, border_radius=11)
                label = f"{name[0]} PINCH" if h.pinching else f"{name[0]} ●"
                txt = f.render(label, True, T.INK)
                surf.blit(txt, txt.get_rect(center=rect.center))
            x -= rect.width + 6
        cap = f.render("HANDS", True, T.INK_FAINT)
        surf.blit(cap, (x - cap.get_width() - 2, y + 5))

    def _draw_effect_bar(self, surf, data: HudData):
        W, H = surf.get_size()
        f = T.font(11, mono=True)
        all_names = self.registry.names
        chip_w, gap = 108, 4
        # Show a window of chips that fits the screen, scrolled to keep the current effect centred.
        fit = max(3, min(len(all_names), (W - 2 * self.MARGIN - 60) // (chip_w + gap)))
        cur = all_names.index(data.current_effect) if data.current_effect in all_names else 0
        first = min(max(0, cur - fit // 2), len(all_names) - fit)
        names = all_names[first:first + fit]
        total = len(names) * chip_w + (len(names) - 1) * gap
        x0 = (W - total) // 2
        more = T.font(14, mono=True)
        yy = H - self.MARGIN - 15
        if first > 0:
            s = more.render(f"‹ {first}", True, T.INK_DIM)
            surf.blit(s, s.get_rect(midright=(x0 - 10, yy)))
        if first + fit < len(all_names):
            s = more.render(f"{len(all_names) - first - fit} ›", True, T.INK_DIM)
            surf.blit(s, s.get_rect(midleft=(x0 + total + 10, yy)))
        y = H - self.MARGIN - 30
        retro_panel(surf, pygame.Rect(x0 - 10, y - 26, total + 20, 64), radius=10)
        self._chip_rects = []
        cap = f"EFFECT  →  {data.effect_target}"
        if not data.effects_enabled:
            cap += "   (FX OFF)"
        cap_s = T.font(10, mono=True).render(cap, True, T.INK_DIM if data.effects_enabled else T.AMBER)
        surf.blit(cap_s, (x0, y - 18))
        hint = T.font(10, mono=True).render(
            f"{cur + 1}/{len(all_names)}   SWIPE OPEN HAND  ·  ← →  ·  1–9 0", True, T.INK_FAINT
        )
        surf.blit(hint, (x0 + total - hint.get_width(), y - 18))
        for i, name in enumerate(names):
            eff = self.registry.get(name)
            number = first + i + 1
            rect = pygame.Rect(x0 + i * (chip_w + gap), y, chip_w, 30)
            self._chip_rects.append((rect, name))
            current = name == data.current_effect
            hover = name == self._hover_chip
            bg_alpha = 0.85 if current else (0.75 if hover else 0.62)
            pygame.draw.rect(surf, T.with_alpha(T.PANEL, bg_alpha), rect, border_radius=3)
            if current:
                pygame.draw.rect(surf, T.with_alpha(eff.accent, 0.9), rect, 1, border_radius=3)
                pygame.draw.line(surf, eff.accent, (rect.x + 8, rect.bottom - 2), (rect.right - 8, rect.bottom - 2), 2)
            num = T.font(9, mono=True).render(str(number), True, eff.accent if current else T.INK_FAINT)
            surf.blit(num, (rect.x + 6, rect.y + 4))
            label = eff.label.upper()
            color = T.INK if current or hover else T.INK_DIM
            txt = f.render(label, True, color)
            if txt.get_width() > chip_w - 18:
                txt = T.font(9, mono=True).render(label, True, color)
            surf.blit(txt, txt.get_rect(midleft=(rect.x + 16, rect.centery + 2)))

    def _draw_notices(self, surf, data: HudData):
        W = surf.get_width()
        y = self.MARGIN + 62  # below the mode tabs
        for n in reversed(data.notices):
            age = data.now - n.time
            if age > 1.8:
                continue
            a = min(1.0, age / 0.15) * min(1.0, (1.8 - age) / 0.5)
            color = T.AMBER if n.kind == "warn" else T.CYAN
            txt = spaced(n.text, 12, T.INK, 2, mono=True)
            rect = txt.get_rect(midtop=(W // 2, y))
            bg = rect.inflate(28, 12)
            pygame.draw.rect(surf, T.with_alpha(T.PANEL, 0.75 * a), bg, border_radius=3)
            pygame.draw.line(surf, T.with_alpha(color, a), bg.bottomleft, bg.bottomright, 1)
            txt.set_alpha(int(255 * a))
            surf.blit(txt, rect)
            y += bg.height + 6
            break  # only the most recent toast; keeps the top clean
