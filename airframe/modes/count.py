"""COUNT — a calculator you operate by showing numbers with your fingers.

* Each hand shows 0–5 (raised fingers). Left hand on screen = first operand,
  right hand = second. The equation is evaluated live.
* 👌 (OK sign: thumb + index touching, other three fingers up): next operation
* Hold the same equation steady for a moment: it is recorded on the tape (Σ)
* Keys: + - * / or ← → operation · Z undo · Del clear tape

The V-sign mode switch is disabled here (V is simply "2"); use Tab or the tabs.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import pygame

from airframe.gestures import metrics
from airframe.rendering import theme as T
from airframe.rendering.glow import GlowLayer
from airframe.rendering.viewport import Viewport

from .base import Experience, Pointer

OPS = ["+", "−", "×", "÷"]
COMMIT_SECONDS = 1.5
OK_FRAMES = 3


class FingerCounter:
    """Majority vote over recent frames + a switching threshold, so the number never flickers."""

    def __init__(self, window: int = 9, switch: int = 6):
        self.history: deque[int] = deque(maxlen=window)
        self.switch = switch
        self.value: int | None = None

    def push(self, n: int) -> int:
        self.history.append(n)
        best, votes = Counter(self.history).most_common(1)[0]
        if self.value is None:
            if votes >= min(self.switch, self.history.maxlen) or len(self.history) >= 3 and votes == len(self.history):
                self.value = best
        elif best != self.value and votes >= self.switch:
            self.value = best
        return self.value if self.value is not None else best


def evaluate(a: int, op: str, b: int) -> float | None:
    if op == "+":
        return a + b
    if op == "−":
        return a - b
    if op == "×":
        return a * b
    if op == "÷":
        return None if b == 0 else a / b
    raise ValueError(op)


def fmt(v: float | None) -> str:
    if v is None:
        return "∞"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


@dataclass
class TapeEntry:
    text: str
    value: float | None
    time: float


@dataclass
class _HandView:
    pid: int
    count: int
    fingers: list[bool]
    tips: np.ndarray
    palm: np.ndarray
    palm_size: float
    color: tuple[int, int, int]
    ok: bool


class CountMode(Experience):
    name, label, accent = "count", "Count", T.AMBER
    allows_mode_gesture = False
    hints = [
        ("SHOW FINGERS", "numbers 0–5"),
        ("OK SIGN", "next operation"),
        ("HOLD STILL", "save to tape"),
        ("TAB", "next mode"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.op_index = 0
        self.counters: dict[int, FingerCounter] = {}
        self.ok_frames: dict[int, int] = {}
        self.views: list[_HandView] = []
        self.tape: list[TapeEntry] = []
        self._stable_key: tuple | None = None
        self._stable_since = 0.0
        self._committed_key: tuple | None = None
        self.commit_progress = 0.0
        self.flash_at = -10.0
        self.op_changed_at = -10.0
        self.now = 0.0

    @property
    def op(self) -> str:
        return OPS[self.op_index]

    def cycle_op(self, d: int = 1) -> str:
        self.op_index = (self.op_index + d) % len(OPS)
        self.op_changed_at = self.now
        return f"OPERATION  {self.op}"

    # -- equation ---------------------------------------------------------------------------
    @property
    def operands(self) -> list[int]:
        return [v.count for v in self.views]

    def equation(self) -> tuple[str, float | None] | None:
        ops = self.operands
        if not ops:
            return None
        if len(ops) == 1:
            return f"{ops[0]}", float(ops[0])
        a, b = ops
        v = evaluate(a, self.op, b)
        return f"{a} {self.op} {b}", v

    @property
    def total(self) -> float:
        return sum(e.value for e in self.tape if e.value is not None)

    # -- update -----------------------------------------------------------------------------
    def update(self, pointers: list[Pointer], dt: float, now: float, size):
        super().update(pointers, dt, now, size)
        self.now = now
        hands = [p for p in pointers if p.hand is not None and not p.vanished]
        hands = sorted(hands, key=lambda p: p.palm[0])[:2]  # left of the screen = first operand
        live = {p.pid for p in hands}
        self.counters = {k: v for k, v in self.counters.items() if k in live}
        views = []
        for p in hands:
            lm = p.hand.landmarks
            counter = self.counters.setdefault(p.pid, FingerCounter())
            ok = metrics.is_ok_sign(lm)
            frames = self.ok_frames.get(p.pid, 0)
            if ok:
                frames += 1
                if frames == OK_FRAMES:
                    self.cycle_op(1)
            else:
                frames = 0
            self.ok_frames[p.pid] = frames
            fingers = metrics.finger_states(lm)
            # while making the OK sign (or pinching) the count is frozen: those poses aren't numbers
            if ok or p.pinching:
                count = counter.value if counter.value is not None else sum(fingers)
            else:
                count = counter.push(sum(fingers))
            views.append(_HandView(p.pid, count, fingers, p.tips, p.palm, p.palm_size, p.color, ok))
        self.views = views
        self._track_commit(now)

    def _track_commit(self, now: float) -> None:
        eq = self.equation()
        key = (eq[0], len(self.views)) if eq and len(self.views) == 2 else None
        if key != self._stable_key:
            self._stable_key = key
            self._stable_since = now
            self._committed_key = None  # a changed equation can be saved again later
        if key is None:
            self.commit_progress = 0.0
            return
        held = now - self._stable_since
        if key == self._committed_key:
            self.commit_progress = 1.0
            return
        self.commit_progress = min(1.0, held / COMMIT_SECONDS)
        if held >= COMMIT_SECONDS:
            text, value = eq
            self.tape.append(TapeEntry(f"{text} = {fmt(value)}", value, now))
            del self.tape[:-8]
            self._committed_key = key
            self.flash_at = now

    def handle_key(self, key, mod):
        mapping = {
            pygame.K_PLUS: 0, pygame.K_KP_PLUS: 0, pygame.K_MINUS: 1, pygame.K_KP_MINUS: 1,
            pygame.K_ASTERISK: 2, pygame.K_KP_MULTIPLY: 2, pygame.K_SLASH: 3, pygame.K_KP_DIVIDE: 3,
        }
        if key == pygame.K_EQUALS and mod & pygame.KMOD_SHIFT:
            key = pygame.K_PLUS
        if key == pygame.K_8 and mod & pygame.KMOD_SHIFT:
            key = pygame.K_ASTERISK
        if key in mapping:
            self.op_index = mapping[key]
            self.op_changed_at = self.now
            return f"OPERATION  {self.op}"
        if key in (pygame.K_LEFT, pygame.K_RIGHT):
            return self.cycle_op(1 if key == pygame.K_RIGHT else -1)
        return None

    def undo(self):
        if self.tape:
            self.tape.pop()
            self._committed_key = None
            return "UNDO"
        return None

    def clear(self):
        self.tape.clear()
        self._committed_key = None
        return "TAPE CLEARED"

    # -- rendering --------------------------------------------------------------------------
    def render_light(self, glow: GlowLayer, now: float) -> None:
        for v in self.views:
            for up, tip in zip(v.fingers, v.tips):
                if up:
                    glow.circle(tip, 9, v.color, -1)
                    glow.circle(tip, 16, v.color, 2)

    def render_overlay(self, layer: pygame.Surface, vp: Viewport, now: float) -> None:
        W, H = layer.get_size()
        hand_rgb = []
        for v in self.views:
            rgb = (v.color[2], v.color[1], v.color[0])
            hand_rgb.append(rgb)
            top = v.tips[:, 1].min()
            anchor = vp.to_screen((v.palm[0], top - v.palm_size * 0.55))
            big = T.font(64, bold=True).render(str(v.count), True, rgb)
            rect = big.get_rect(midbottom=(anchor[0], anchor[1]))
            ring = rect.inflate(30, 10)
            pygame.draw.rect(layer, T.with_alpha(T.PANEL, 0.55), ring, border_radius=12)
            pygame.draw.rect(layer, T.with_alpha(rgb, 0.8), ring, 1, border_radius=12)
            layer.blit(big, rect)
            if v.ok:
                tag = T.font(11, mono=True, bold=True).render(f"OK → {self.op}", True, T.AMBER)
                layer.blit(tag, tag.get_rect(midbottom=(ring.centerx, ring.top - 6)))

        # live equation
        eq = self.equation()
        cy = 128
        if eq is None:
            msg = T.font(16).render("Show your hands — each one is a number from 0 to 5", True, T.INK_DIM)
            layer.blit(msg, msg.get_rect(center=(W // 2, cy + 20)))
        else:
            parts: list[tuple[str, tuple[int, int, int]]] = []
            ops = self.operands
            parts.append((str(ops[0]), hand_rgb[0]))
            if len(ops) == 2:
                op_flash = max(0.0, 1 - (now - self.op_changed_at) / 0.6)
                parts.append((self.op, T.mix(T.AMBER, T.INK, 1 - op_flash) if op_flash else T.AMBER))
                parts.append((str(ops[1]), hand_rgb[1]))
            parts.append(("=", T.INK_DIM))
            parts.append((fmt(eq[1]), T.INK))
            font = T.font(56, bold=True)
            rendered = [font.render(t, True, c) for t, c in parts]
            gap = 26
            total = sum(r.get_width() for r in rendered) + gap * (len(rendered) - 1)
            panel = pygame.Rect(0, 0, total + 70, 92)
            panel.midtop = (W // 2, cy - 30)
            flash = max(0.0, 1 - (now - self.flash_at) / 0.7)
            pygame.draw.rect(layer, T.with_alpha(T.PANEL, 0.72), panel, border_radius=10)
            pygame.draw.rect(layer, T.with_alpha(T.mix(T.INK_FAINT, T.GREEN, flash), 1.0), panel, 1, border_radius=10)
            x = panel.x + 35
            for r in rendered:
                layer.blit(r, r.get_rect(midleft=(x, panel.centery)))
                x += r.get_width() + gap
            if len(ops) == 2:  # hold-to-save progress
                bar = pygame.Rect(panel.x + 24, panel.bottom - 8, panel.width - 48, 3)
                pygame.draw.rect(layer, T.with_alpha(T.INK_FAINT, 0.6), bar)
                fill = bar.copy()
                fill.width = int(bar.width * self.commit_progress)
                pygame.draw.rect(layer, T.GREEN if self.commit_progress >= 1 else T.AMBER, fill)
            ops_row = "   ".join(f"[{o}]" if o == self.op else f" {o} " for o in OPS)
            s = T.font(12, mono=True).render(ops_row, True, T.INK_DIM)
            sr = s.get_rect(midtop=(W // 2, panel.bottom + 10))
            pygame.draw.rect(layer, T.with_alpha(T.PANEL, 0.72), sr.inflate(18, 8), border_radius=6)
            layer.blit(s, sr)

        # tape
        if self.tape:
            f = T.font(13, mono=True)
            x = W - 230
            y = 160
            box = pygame.Rect(x - 16, y - 34, 228, 40 + 22 * len(self.tape) + 34)
            pygame.draw.rect(layer, T.with_alpha(T.PANEL, 0.75), box, border_radius=6)
            pygame.draw.line(layer, T.AMBER, (box.x + 14, box.y), (box.x + 60, box.y), 2)
            layer.blit(T.font(10, mono=True).render("TAPE", True, T.INK_FAINT), (x, y - 24))
            for i, e in enumerate(self.tape):
                fresh = max(0.0, 1 - (now - e.time) / 0.8)
                s = f.render(e.text, True, T.mix(T.INK_DIM, T.GREEN, fresh))
                layer.blit(s, (x, y + i * 22))
            y += len(self.tape) * 22 + 6
            pygame.draw.line(layer, T.INK_FAINT, (x, y), (box.right - 16, y))
            s = T.font(15, mono=True, bold=True).render(f"Σ  {fmt(self.total)}", True, T.INK)
            layer.blit(s, (x, y + 8))
