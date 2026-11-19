"""Calibration panel: live pinch meter + sliders for gesture tolerances."""

from __future__ import annotations

from typing import Callable

import pygame

from airframe.config import GestureConfig
from airframe.gestures.autocalib import AutoCalibration
from airframe.rendering import theme as T

from .widgets import Button, Slider


class CalibrationPanel:
    WIDTH = 310
    HEIGHT = 385
    METER_MAX = 1.2

    def __init__(
        self,
        config: GestureConfig,
        on_change: Callable[[], None],
        on_save: Callable[[], None],
        on_message: Callable[[str, str], None] = lambda text, kind: None,
    ):
        self.config = config
        self.on_change = on_change
        self.on_save = on_save
        self.on_message = on_message
        self.auto: AutoCalibration | None = None
        self._now = 0.0
        self.visible = False
        self.focus = 0
        self._defaults = GestureConfig()
        c = config

        def setter(attr):
            def _set(v):
                setattr(c, attr, float(v))
                if attr == "pinch_on" and c.pinch_off < c.pinch_on + 0.04:
                    c.pinch_off = round(c.pinch_on + 0.04, 2)
                if attr == "pinch_off" and c.pinch_off < c.pinch_on + 0.04:
                    c.pinch_on = round(c.pinch_off - 0.04, 2)
                self.on_change()
            return _set

        self.sliders = [
            Slider("PINCH ENGAGE", 0.10, 0.50, lambda: c.pinch_on, setter("pinch_on"), step=0.01),
            Slider("PINCH RELEASE", 0.15, 0.70, lambda: c.pinch_off, setter("pinch_off"), step=0.01),
            Slider("SMOOTHING (MIN CUTOFF)", 0.2, 4.0, lambda: c.filter_min_cutoff, setter("filter_min_cutoff"), step=0.1, fmt="{:.1f} Hz"),
            Slider("RESPONSIVENESS (BETA)", 0.0, 0.08, lambda: c.filter_beta, setter("filter_beta"), step=0.001, fmt="{:.3f}"),
            Slider("SWIPE DISTANCE", 0.10, 0.60, lambda: c.swipe_min_distance, setter("swipe_min_distance"), step=0.01, fmt="{:.0%}"),
        ]
        self.buttons = [
            Button("AUTO", self.start_auto, is_active=lambda: self.auto is not None),
            Button("SWIPE ON", self._toggle_swipe, is_active=lambda: c.swipe_enabled),
            Button("RESET", self._reset),
            Button("SAVE", on_save),
        ]
        self.rect = pygame.Rect(0, 0, 0, 0)

    def _toggle_swipe(self):
        self.config.swipe_enabled = not self.config.swipe_enabled
        self.buttons[1].label = "SWIPE ON" if self.config.swipe_enabled else "SWIPE OFF"
        self.on_change()

    def _reset(self):
        for k in ("pinch_on", "pinch_off", "filter_min_cutoff", "filter_beta", "swipe_min_distance", "swipe_enabled"):
            setattr(self.config, k, getattr(self._defaults, k))
        self.buttons[1].label = "SWIPE ON"
        self.on_change()

    # -- auto calibration ------------------------------------------------------------------
    def start_auto(self) -> None:
        self.visible = True
        self.auto = AutoCalibration(self._now)

    def tick(self, now: float, ratios: list[float]) -> None:
        """Called every frame with the live pinch ratios of the visible hands."""
        self._now = now
        if self.auto is None:
            return
        phase = self.auto.update(now, ratios)
        if phase == "done" and self.auto.result:
            on, off = self.auto.result
            self.config.pinch_on, self.config.pinch_off = on, off
            self.on_change()
            self.on_save()
            self.on_message(f"PINCH CALIBRATED  {on:.2f} / {off:.2f}", "info")
            self.auto = None
        elif phase == "failed":
            self.on_message("CALIBRATION FAILED — KEEP ONE HAND VISIBLE", "warn")
            self.auto = None

    def draw_auto(self, surf: pygame.Surface) -> None:
        """Big centred instruction while auto calibration runs."""
        a = self.auto
        if a is None:
            return
        W, H = surf.get_size()
        box = pygame.Rect(0, 0, 460, 132)
        box.center = (W // 2 - (self.WIDTH // 2 if self.visible else 0), H // 2)
        pygame.draw.rect(surf, T.with_alpha(T.PANEL, 0.9), box, border_radius=8)
        color = T.CYAN if a.phase == "open" else T.GREEN
        pygame.draw.rect(surf, color, box, 1, border_radius=8)
        step = "1/2" if a.phase == "open" else "2/2"
        title = "OPEN YOUR HAND" if a.phase == "open" else "PINCH  ·  TOUCH THUMB AND INDEX"
        sub = "Fingers spread, facing the camera" if a.phase == "open" else "Hold it like you mean it — firmly touching"
        t = T.font(10, mono=True).render(f"AUTO CALIBRATION  {step}", True, T.INK_DIM)
        surf.blit(t, (box.x + 22, box.y + 18))
        surf.blit(T.font(18, bold=True).render(title, True, T.INK), (box.x + 22, box.y + 38))
        surf.blit(T.font(12).render(sub, True, T.INK_DIM), (box.x + 22, box.y + 68))
        bar = pygame.Rect(box.x + 22, box.bottom - 26, box.width - 44, 4)
        pygame.draw.rect(surf, T.with_alpha(T.INK_FAINT, 1.0), bar, border_radius=2)
        fill = bar.copy()
        fill.width = int(bar.width * a.progress(self._now))
        pygame.draw.rect(surf, color, fill, border_radius=2)

    # -- input ----------------------------------------------------------------------------
    def handle(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False
        for s in self.sliders:
            if s.handle(event):
                return True
        for b in self.buttons:
            if b.handle(event):
                return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.focus = (self.focus - 1) % len(self.sliders)
                return True
            if event.key == pygame.K_DOWN:
                self.focus = (self.focus + 1) % len(self.sliders)
                return True
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.sliders[self.focus].nudge(1 if event.key == pygame.K_RIGHT else -1)
                return True
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP) and self.rect.collidepoint(event.pos):
            return True  # swallow clicks on the panel background
        return False

    def hit(self, pos) -> bool:
        return self.visible and self.rect.collidepoint(pos)

    # -- drawing --------------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, hand_ratios: list[tuple[str, float, bool]]) -> None:
        if not self.visible:
            return
        W, H = surf.get_size()
        m = 18
        top = 120
        self.rect = pygame.Rect(W - self.WIDTH - m, top, self.WIDTH, self.HEIGHT)
        r = self.rect
        pygame.draw.rect(surf, T.with_alpha(T.PANEL, 0.86), r, border_radius=6)
        pygame.draw.rect(surf, T.with_alpha(T.INK_FAINT, 1.0), r, 1, border_radius=6)
        pygame.draw.line(surf, T.CYAN, (r.x + 16, r.y), (r.x + 64, r.y), 2)

        x, y = r.x + 16, r.y + 14
        surf.blit(T.font(12, mono=True, bold=True).render("CALIBRATION", True, T.INK), (x, y))
        y += 22
        surf.blit(T.font(10, mono=True).render("Pinch ratio = thumb↔index / palm", True, T.INK_FAINT), (x, y))
        y += 20

        # live meter
        meter = pygame.Rect(x, y, r.width - 32, 14)
        pygame.draw.rect(surf, T.with_alpha(T.BG, 0.9), meter, border_radius=3)
        on_x = meter.x + meter.width * self.config.pinch_on / self.METER_MAX
        off_x = meter.x + meter.width * self.config.pinch_off / self.METER_MAX
        zone = pygame.Rect(meter.x, meter.y, on_x - meter.x, meter.height)
        pygame.draw.rect(surf, T.with_alpha(T.GREEN, 0.25), zone, border_radius=3)
        hyst = pygame.Rect(on_x, meter.y, off_x - on_x, meter.height)
        pygame.draw.rect(surf, T.with_alpha(T.AMBER, 0.18), hyst)
        for hx, col in ((on_x, T.GREEN), (off_x, T.AMBER)):
            pygame.draw.line(surf, col, (hx, meter.y - 3), (hx, meter.bottom + 2), 1)
        for name, ratio, pinching in hand_ratios:
            px = meter.x + meter.width * min(ratio, self.METER_MAX) / self.METER_MAX
            col = T.HAND_COLORS.get(name, T.BLUE)
            pygame.draw.line(surf, col, (px, meter.y + 1), (px, meter.bottom - 1), 3)
            lbl = T.font(9, mono=True).render(f"{name[0]} {ratio:.2f}{' ●' if pinching else ''}", True, col)
            surf.blit(lbl, (min(px, meter.right - lbl.get_width()), meter.bottom + 3 + (12 if name == "Left" else 0)))
        if not hand_ratios:
            msg = T.font(9, mono=True).render("show a hand to see its live value", True, T.INK_FAINT)
            surf.blit(msg, (meter.x, meter.bottom + 4))
        y = meter.bottom + 34

        for i, s in enumerate(self.sliders):
            s.rect = pygame.Rect(x, y, r.width - 32, 30)
            s.draw(surf, focused=(i == self.focus))
            y += 44

        bw = (r.width - 32 - 18) // 4
        for i, b in enumerate(self.buttons):
            b.rect = pygame.Rect(x + i * (bw + 6), y + 2, bw, 24)
            b.draw(surf)
        y += 36
        hint = T.font(9, mono=True).render("A auto · ↑↓ select · ←→ adjust · C close", True, T.INK_FAINT)
        surf.blit(hint, (x, min(y, r.bottom - 16)))
