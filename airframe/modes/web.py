"""WEB — elastic threads spun between your fingertips.

* Touch two fingertips together (across both hands, or the thumb with any
  finger of the same hand) and pull them apart: a glowing thread forms.
* Pull slowly: it stretches like melted cheese. Yank it: it SNAPS.
* Bring the hands closer: the thread sags under gravity.
* Free fingertips (and the mouse) push threads and make them wobble.
* G: gravity on/off · Del: cut everything

Each thread is a Verlet rope (vectorised) whose ends are pinned to fingertips.
Its rest length "creeps" toward the current length, so slow pulls extend the
thread; the snap happens when the extension beyond rest length is too large,
which in practice means pulling fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from airframe.rendering import theme as T
from airframe.rendering.glow import GlowLayer
from airframe.rendering.viewport import Viewport

from .base import MOUSE_POINTER, Experience, Pointer

N = 18                    # particles per thread
ITERATIONS = 14           # constraint relaxation passes
GRAVITY = 900.0
CREEP = 3.0               # 1/s — how fast rest length follows the pulled length
MAX_THREADS = 30
COOLDOWN = 0.5            # s before the same fingertip pair can re-form a thread
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")

Anchor = tuple[int, int]  # (pointer id, tip index)


@dataclass(eq=False)
class Thread:
    a: Anchor
    b: Anchor
    pos: np.ndarray
    prev: np.ndarray
    rest: float
    born: float
    pin_a: bool = True
    pin_b: bool = True
    tension: float = 0.0          # 0..1 of the snap threshold
    dying_since: float | None = None

    @property
    def key(self) -> frozenset:
        return frozenset((self.a, self.b))

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.pos, axis=0), axis=1).sum())


def make_thread(a: Anchor, b: Anchor, pa, pb, now: float, rest: float | None = None) -> Thread:
    pts = np.linspace(np.asarray(pa, float), np.asarray(pb, float), N)
    span = float(np.linalg.norm(np.asarray(pb, float) - np.asarray(pa, float)))
    return Thread(a, b, pts, pts.copy(), rest if rest is not None else max(span, 8.0), now)


def simulate(t: Thread, pa, pb, dt: float, gravity: float) -> None:
    """One Verlet step with pinned ends (``pa``/``pb`` may be None when unpinned)."""
    vel = (t.pos - t.prev) * 0.985
    t.prev = t.pos.copy()
    t.pos = t.pos + vel
    t.pos[:, 1] += gravity * dt * dt
    seg = t.rest / (N - 1)
    for _ in range(ITERATIONS):
        if t.pin_a and pa is not None:
            t.pos[0] = pa
        if t.pin_b and pb is not None:
            t.pos[-1] = pb
        d = t.pos[1:] - t.pos[:-1]
        length = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        corr = d * ((length - seg) / length) * 0.5
        w_a = np.ones((N - 1, 1))
        w_b = np.ones((N - 1, 1))
        if t.pin_a and pa is not None:   # pinned point doesn't move: neighbour takes the full correction
            w_a[0], w_b[0] = 0.0, 2.0
        if t.pin_b and pb is not None:
            w_a[-1], w_b[-1] = 2.0, 0.0
        t.pos[:-1] += corr * w_a
        t.pos[1:] -= corr * w_b
    if t.pin_a and pa is not None:
        t.pos[0] = pa
    if t.pin_b and pb is not None:
        t.pos[-1] = pb


class WebMode(Experience):
    name, label, accent = "web", "Web", T.VIOLET
    hints = [
        ("TOUCH FINGERTIPS", "spin a thread"),
        ("PULL SLOWLY", "stretch"),
        ("YANK", "snap"),
        ("G  DEL", "gravity · cut all"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[Thread] = []
        self.anchors: dict[Anchor, np.ndarray] = {}
        self.cooldown: dict[frozenset, float] = {}
        self.sparks: list[tuple[np.ndarray, float]] = []
        self.gravity = True
        self.now = 0.0
        self.touch_points: list[np.ndarray] = []
        self.pushers: list[np.ndarray] = []

    def snap_extension(self, palm: float = 90.0) -> float:
        return max(120.0, palm * 1.8)

    # -- update -----------------------------------------------------------------------------
    def update(self, pointers: list[Pointer], dt: float, now: float, size):
        super().update(pointers, dt, now, size)
        self.now = now
        dt = min(dt, 1 / 20)
        hands = [p for p in pointers if p.hand is not None and not p.vanished]
        self.anchors = {(p.pid, i): np.asarray(tip, float) for p in hands for i, tip in enumerate(p.tips)}
        palm = float(np.mean([p.palm_size for p in hands])) if hands else 90.0

        self._form_threads(hands, now)

        attached = {a for t in self.threads if t.dying_since is None for a in (t.a, t.b)}
        self.pushers = [pos for key, pos in self.anchors.items() if key not in attached]
        self.pushers += [p.pos for p in pointers if p.pid == MOUSE_POINTER and p.pinching]

        g = GRAVITY if self.gravity else 0.0
        limit = self.snap_extension(palm)
        max_len = 0.95 * max(size)
        for t in self.threads:
            pa, pb = self.anchors.get(t.a), self.anchors.get(t.b)
            if t.dying_since is None:
                if pa is None or pb is None:  # a hand left: the thread drops loose
                    self._snap(t, now, at=None)
                else:
                    span = float(np.linalg.norm(pb - pa))
                    # rest length creeps toward the span; shrinks slower so slack sags nicely
                    rate = CREEP if span > t.rest else CREEP * 0.35
                    t.rest += (span - t.rest) * (1 - math.exp(-rate * dt))
                    extension = span - t.rest
                    t.tension = float(np.clip(extension / limit, 0, 1))
                    if extension > limit or span > max_len:
                        self._snap(t, now, at=(pa + pb) / 2)
            simulate(t, pa, pb, dt, g)
            self._push(t)
        self.threads = [t for t in self.threads if t.dying_since is None or now - t.dying_since < 1.6]
        self.sparks = [s for s in self.sparks if now - s[1] < 0.5]
        self.cooldown = {k: v for k, v in self.cooldown.items() if v > now}

    def _form_threads(self, hands: list[Pointer], now: float) -> None:
        self.touch_points = []
        live = {t.key for t in self.threads if t.dying_since is None}
        for hi, h1 in enumerate(hands):
            for h2 in hands[hi:]:
                same = h1.pid == h2.pid
                touch = max(14.0, 0.2 * min(h1.palm_size, h2.palm_size))
                for i, ti in enumerate(h1.tips):
                    for j, tj in enumerate(h2.tips):
                        if same and (j <= i or i != 0):  # same hand: only thumb ↔ other fingers
                            continue
                        if float(np.linalg.norm(ti - tj)) > touch:
                            continue
                        a, b = (h1.pid, i), (h2.pid, j)
                        key = frozenset((a, b))
                        self.touch_points.append((ti + tj) / 2)
                        if key in live or self.cooldown.get(key, 0) > now:
                            continue
                        if sum(1 for t in self.threads if t.dying_since is None) >= MAX_THREADS:
                            return
                        self.threads.append(make_thread(a, b, ti, tj, now))
                        live.add(key)
                        self.cooldown[key] = now + COOLDOWN

    def _snap(self, t: Thread, now: float, at) -> None:
        """Break ``t``: it becomes two loose halves, each still hanging from its fingertip."""
        t.dying_since = now
        self.cooldown[t.key] = now + COOLDOWN
        if at is None:
            t.pin_a = t.a in self.anchors
            t.pin_b = t.b in self.anchors
            return
        k = N // 2
        half_b = Thread(t.b, t.a, t.pos[k:][::-1].copy(), t.prev[k:][::-1].copy(), t.rest, t.born, True, False)
        half_b.dying_since = now
        # recoil: give the free ends a kick back toward their fingertip
        t.pos, t.prev = np.vstack([t.pos[:k], t.pos[k - 1:k]]), np.vstack([t.prev[:k], t.prev[k - 1:k]])
        t.pos = np.linspace(t.pos[0], t.pos[-1], N)
        t.prev = t.pos + (t.pos - t.pos[0]) * 0.08
        half_b.pos = np.linspace(half_b.pos[0], half_b.pos[-1], N)
        half_b.prev = half_b.pos + (half_b.pos - half_b.pos[0]) * 0.08
        t.pin_b = False
        t.rest = half_b.rest = t.rest * 0.5
        self.threads.append(half_b)
        self.sparks.append((np.asarray(at, float), now))

    def _push(self, t: Thread) -> None:
        for p in self.pushers:
            d = t.pos - p
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            inside = dist < 22.0
            if inside.any():
                t.pos = np.where(inside, p + d / dist * 22.0, t.pos)

    def handle_key(self, key, mod):
        if key == pygame.K_g:
            self.gravity = not self.gravity
            return "GRAVITY ON" if self.gravity else "ZERO GRAVITY"
        return None

    def clear(self):
        now = self.now
        for t in self.threads:
            if t.dying_since is None:
                self._snap(t, now, at=t.pos[N // 2])
        return "WEB CUT"

    # -- rendering --------------------------------------------------------------------------
    def render_light(self, glow: GlowLayer, now: float) -> None:
        for t in self.threads:
            fade = 1.0 if t.dying_since is None else max(0.0, 1 - (now - t.dying_since) / 1.6)
            if fade <= 0:
                continue
            k = t.tension
            # silk white-cyan → violet → hot magenta as the tension rises
            base = np.array([255, 235, 200]) * (1 - k) + np.array([255, 80, 230]) * k
            col = tuple(int(c * fade) for c in base)
            width = 2.2 - 1.0 * k
            glow.polyline(t.pos, col, width)
            if t.dying_since is None and k > 0.6 and int(now * 20) % 2:  # flicker before snapping
                glow.polyline(t.pos, (255, 255, 255), 1)
        for a in {x for t in self.threads if t.dying_since is None for x in (t.a, t.b)}:
            if a in self.anchors:
                glow.circle(self.anchors[a], 5, (255, 220, 255), -1)
        for pos, start in self.sparks:
            age = (now - start) / 0.5
            col = tuple(int(c * (1 - age)) for c in (200, 120, 255))
            glow.circle(pos, 6 + 60 * age, col, 2)
            for i in range(8):
                ang = i / 8 * math.tau + start
                r0, r1 = 10 + 40 * age, 18 + 70 * age
                glow.line(pos + r0 * np.array([math.cos(ang), math.sin(ang)]),
                          pos + r1 * np.array([math.cos(ang), math.sin(ang)]), col, 2)

    def render_overlay(self, layer: pygame.Surface, vp: Viewport, now: float) -> None:
        for p in self.touch_points:
            pygame.draw.circle(layer, T.with_alpha(T.INK, 0.8), vp.to_screen(p), 9, 1)
        for pos in self.pushers:
            pygame.draw.circle(layer, T.with_alpha(T.INK_DIM, 0.5), vp.to_screen(pos), 3)
        alive = [t for t in self.threads if t.dying_since is None]
        f = T.font(10, mono=True)
        s = f.render(f"{len(alive):2d} THREADS  ·  GRAVITY {'ON' if self.gravity else 'OFF'}", True, T.INK_FAINT)
        layer.blit(s, (layer.get_width() - s.get_width() - 18, 130))
        tense = max(alive, key=lambda th: th.tension, default=None)
        for t in [tense] if tense is not None else []:  # tension meter on the most stretched thread only
            if t.tension > 0.35:
                mid = vp.to_screen(t.pos[N // 2])
                txt = f.render(f"{t.tension * 100:3.0f}%", True, T.mix(T.INK, T.MAGENTA, t.tension))
                layer.blit(txt, txt.get_rect(midbottom=(mid[0], mid[1] - 10)))
