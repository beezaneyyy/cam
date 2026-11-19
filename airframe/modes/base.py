"""Common interface for interaction modes ("experiences").

Each mode receives a uniform list of ``Pointer`` objects every frame — tracked
hands and the mouse look the same — so modes only deal with "where is it, is it
pinching, is the hand open, how fast is it moving".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pygame

from airframe.gestures.engine import HandInfo
from airframe.gestures.metrics import palm_size
from airframe.rendering import theme as T
from airframe.rendering.glow import GlowLayer
from airframe.rendering.viewport import Viewport

MOUSE_POINTER = 0

# BGR colours used by the light layer
CYAN_BGR = (255, 230, 40)
MAGENTA_BGR = (220, 60, 255)
VIOLET_BGR = (255, 100, 160)
HAND_BGR = {"Left": VIOLET_BGR, "Right": CYAN_BGR}


@dataclass
class Pointer:
    pid: int
    pos: np.ndarray                 # interaction point (pinch point), camera px
    pinching: bool
    is_open: bool = False
    palm: np.ndarray | None = None
    palm_size: float = 80.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hand: HandInfo | None = None
    handedness: str = "Right"
    started: bool = False           # pinch began this frame
    ended: bool = False             # pinch ended this frame (or the pointer vanished while pinching)
    vanished: bool = False

    @property
    def tips(self) -> np.ndarray:
        """Fingertip positions (5, 2) — just the pointer position for the mouse."""
        if self.hand is None:
            return self.pos[None, :]
        return self.hand.landmarks[[4, 8, 12, 16, 20], :2]

    @property
    def color(self) -> tuple[int, int, int]:
        return HAND_BGR.get(self.handedness, CYAN_BGR)


class PointerTracker:
    """Turns hands + mouse into pointers with velocity and pinch edge flags."""

    def __init__(self, velocity_smoothing: float = 0.5):
        self.k = velocity_smoothing
        self._prev: dict[int, tuple[np.ndarray, float, bool]] = {}
        self._vel: dict[int, np.ndarray] = {}
        self._last: dict[int, Pointer] = {}

    def update(self, hands: list[HandInfo], now: float, mouse: tuple[np.ndarray, bool] | None = None) -> list[Pointer]:
        out: list[Pointer] = []
        for h in hands:
            out.append(
                Pointer(
                    pid=h.hand_id,
                    pos=np.asarray(h.interaction_point, np.float64),
                    pinching=h.pinching,
                    is_open=h.is_open,
                    palm=np.asarray(h.palm_center, np.float64),
                    palm_size=palm_size(h.landmarks),
                    hand=h,
                    handedness=h.handedness,
                )
            )
        if mouse is not None:
            pos, down = mouse
            out.append(Pointer(MOUSE_POINTER, np.asarray(pos, np.float64), down, palm=np.asarray(pos, np.float64)))

        seen = set()
        for p in out:
            seen.add(p.pid)
            prev = self._prev.get(p.pid)
            if prev is not None:
                ppos, pt, ppinch = prev
                dt = now - pt
                if dt > 1e-4:
                    inst = (p.pos - ppos) / dt
                    v = self._vel.get(p.pid, inst)
                    self._vel[p.pid] = v * self.k + inst * (1 - self.k)
                p.started = p.pinching and not ppinch
                p.ended = ppinch and not p.pinching
            else:
                p.started = p.pinching
            p.velocity = self._vel.get(p.pid, np.zeros(2)).copy()
            self._prev[p.pid] = (p.pos.copy(), now, p.pinching)
            self._last[p.pid] = p

        for pid in list(self._prev):
            if pid not in seen:
                last = self._last.get(pid)
                if last is not None and self._prev[pid][2]:
                    gone = Pointer(pid, self._prev[pid][0], False, palm=last.palm, palm_size=last.palm_size,
                                   handedness=last.handedness, velocity=self._vel.get(pid, np.zeros(2)).copy())
                    gone.ended = gone.vanished = True
                    out.append(gone)
                self._prev.pop(pid, None)
                self._vel.pop(pid, None)
                self._last.pop(pid, None)
        return out


class Experience:
    """Base class for a mode."""

    name = "base"
    label = "Base"
    accent = T.CYAN
    hints: list[tuple[str, str]] = []
    allows_mode_gesture = True   # False when the V-sign means something in this mode

    def __init__(self) -> None:
        self.size = (1280, 720)

    def enter(self) -> None: ...

    def exit(self) -> None: ...

    def on_camera(self, image: np.ndarray) -> None:
        """Latest raw camera frame (read-only, shared with the tracker — copy before keeping)."""

    def on_gestures(self, gf, now: float) -> None:
        """Raw gesture frame with discrete events, for modes that need more than pointers."""

    def update(self, pointers: list[Pointer], dt: float, now: float, size: tuple[int, int]) -> None:
        self.size = size

    def render_image(self, img: np.ndarray, now: float) -> None:
        """Draw opaque content directly onto the camera-space image (before light)."""

    def render_light(self, glow: GlowLayer, now: float) -> None:
        """Draw additive light (camera space)."""

    def render_overlay(self, layer: pygame.Surface, vp: Viewport, now: float) -> None:
        """Draw crisp vector UI (screen space)."""

    def handle_key(self, key: int, mod: int) -> str | None:
        """Return a notice text if the key did something."""
        return None

    def clear(self) -> str | None:
        return None

    def undo(self) -> str | None:
        return None
