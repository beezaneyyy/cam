"""Minimal immediate-style widgets: buttons and sliders (mouse driven)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pygame

from airframe.rendering import theme as T


@dataclass
class Button:
    label: str
    on_click: Callable[[], None]
    hint: str = ""
    is_active: Callable[[], bool] = field(default=lambda: False)
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    hover: bool = False

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.on_click()
            return True
        return False

    def draw(self, surf: pygame.Surface) -> None:
        active = self.is_active()
        border = T.CYAN if active else (T.INK_DIM if self.hover else T.INK_FAINT)
        bg = T.with_alpha(T.CYAN, 0.14) if active else T.with_alpha(T.PANEL, 0.72 if self.hover else 0.55)
        if T.is_retro():  # chunky handheld button: solid fill, hard shadow, pressed look when active
            border = T.BLUE
            bg = T.mix(T.PANEL, T.CYAN, 0.35) if active else T.PANEL
            pygame.draw.rect(surf, T.with_alpha(T.INK, 0.3), self.rect.move(0, 3), border_radius=6)
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, border, self.rect, 1, border_radius=4)
        f = T.font(11, mono=True)
        txt = f.render(self.label, True, T.INK if active or self.hover else T.INK_DIM)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


@dataclass
class Slider:
    label: str
    minimum: float
    maximum: float
    get: Callable[[], float]
    set: Callable[[float], None]
    fmt: str = "{:.2f}"
    step: float = 0.01
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    dragging: bool = False

    @property
    def track(self) -> pygame.Rect:
        return pygame.Rect(self.rect.x, self.rect.bottom - 8, self.rect.width, 4)

    def _set_from_x(self, x: float) -> None:
        tr = self.track
        frac = min(1.0, max(0.0, (x - tr.x) / max(tr.width, 1)))
        value = self.minimum + frac * (self.maximum - self.minimum)
        if self.step:
            value = round(value / self.step) * self.step
        self.set(value)

    def nudge(self, direction: int) -> None:
        self.set(min(self.maximum, max(self.minimum, self.get() + direction * self.step)))

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit = self.rect.inflate(0, 8)
            if hit.collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was = self.dragging
            self.dragging = False
            return was
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False

    def draw(self, surf: pygame.Surface, focused: bool = False) -> None:
        value = self.get()
        f = T.font(11, mono=True)
        surf.blit(f.render(self.label, True, T.INK if focused else T.INK_DIM), (self.rect.x, self.rect.y))
        val = f.render(self.fmt.format(value), True, T.CYAN)
        surf.blit(val, (self.rect.right - val.get_width(), self.rect.y))
        tr = self.track
        pygame.draw.rect(surf, T.with_alpha(T.INK_FAINT, 0.8), tr, border_radius=2)
        frac = (value - self.minimum) / max(self.maximum - self.minimum, 1e-9)
        fill = tr.copy()
        fill.width = int(tr.width * min(1.0, max(0.0, frac)))
        pygame.draw.rect(surf, T.CYAN, fill, border_radius=2)
        knob = (tr.x + fill.width, tr.centery)
        pygame.draw.circle(surf, T.BG, knob, 7)
        pygame.draw.circle(surf, T.CYAN if focused or self.dragging else T.INK, knob, 6, 2)
