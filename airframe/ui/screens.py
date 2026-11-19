"""Full-screen states and overlays: splash, error, onboarding hint, help."""

from __future__ import annotations

import math

import pygame

from airframe.rendering import theme as T

from .hud import spaced


def _wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        words, cur = para.split(), ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if font.size(trial)[0] <= width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def _grid(surf: pygame.Surface, t: float, alpha: float = 0.05) -> None:
    W, H = surf.get_size()
    step = 48
    off = (t * 8) % step
    c = T.with_alpha(T.BLUE, alpha)
    for x in range(-step, W + step, step):
        pygame.draw.line(surf, c, (x + off, 0), (x + off, H))
    for y in range(-step, H + step, step):
        pygame.draw.line(surf, c, (0, y + off), (W, y + off))


def draw_splash(surf: pygame.Surface, t: float, steps: list[tuple[str, str]], fade: float = 1.0) -> None:
    """``steps``: (label, state) where state ∈ {"pending", "active", "done", "error"}."""
    W, H = surf.get_size()
    layer = pygame.Surface((W, H), pygame.SRCALPHA)
    layer.fill((*T.BG, 255))
    _grid(layer, t, 0.06)
    cx, cy = W // 2, H // 2 - 30

    title = spaced("AIRFRAME", 44, T.INK, 16, bold=True)
    layer.blit(title, title.get_rect(center=(cx, cy)))
    sub = spaced("HAND TRACKING VISUAL EXPERIENCE", 11, T.INK_DIM, 5, mono=True)
    layer.blit(sub, sub.get_rect(center=(cx, cy + 42)))

    # scanning line
    lw = 360
    x0 = cx - lw // 2
    pygame.draw.line(layer, T.with_alpha(T.INK_FAINT, 0.8), (x0, cy + 66), (x0 + lw, cy + 66))
    phase = (t * 0.7) % 1.0
    hx = x0 + phase * lw
    for i in range(30):
        a = (1 - i / 30) * 0.9
        pygame.draw.line(layer, T.with_alpha(T.CYAN, a), (hx - i * 2, cy + 66), (hx - i * 2 - 2, cy + 66), 2)

    f = T.font(12, mono=True)
    y = cy + 96
    for label, state in steps:
        if state == "done":
            mark, col = "✓", T.GREEN
        elif state == "active":
            mark, col = "◌◔◑◕"[int(t * 8) % 4], T.CYAN
        elif state == "error":
            mark, col = "×", T.RED
        else:
            mark, col = "·", T.INK_FAINT
        line = f.render(f"{mark}  {label}", True, col if state != "pending" else T.INK_FAINT)
        layer.blit(line, (cx - 150, y))
        y += 22
    layer.set_alpha(int(255 * max(0.0, min(1.0, fade))))
    surf.blit(layer, (0, 0))


def draw_error(surf: pygame.Surface, t: float, title: str, message: str) -> None:
    W, H = surf.get_size()
    surf.fill(T.BG)
    layer = pygame.Surface((W, H), pygame.SRCALPHA)
    _grid(layer, t, 0.04)
    cx, cy = W // 2, H // 2
    box = pygame.Rect(0, 0, min(620, W - 40), 270)
    box.center = (cx, cy)
    pygame.draw.rect(layer, T.with_alpha(T.PANEL, 0.95), box, border_radius=8)
    pygame.draw.rect(layer, T.with_alpha(T.RED, 0.6), box, 1, border_radius=8)
    pygame.draw.line(layer, T.RED, (box.x + 24, box.y), (box.x + 90, box.y), 2)
    surf.blit(layer, (0, 0))

    pulse = 0.6 + 0.4 * math.sin(t * 3)
    pygame.draw.circle(surf, T.mix(T.BG, T.RED, pulse), (box.x + 34, box.y + 40), 6)
    surf.blit(spaced(title.upper(), 16, T.INK, 4, bold=True), (box.x + 52, box.y + 30))
    f = T.font(14)
    y = box.y + 76
    for line in _wrap(message, f, box.width - 56):
        surf.blit(f.render(line, True, T.INK_DIM), (box.x + 28, y))
        y += 22
    hint = T.font(11, mono=True).render("R  RETRY      ·      Q / ESC  QUIT", True, T.CYAN)
    surf.blit(hint, hint.get_rect(midbottom=(cx, box.bottom - 22)))


def draw_onboarding(surf: pygame.Surface, t: float, alpha: float) -> None:
    """Animated hint: two pinches meeting at opposite corners draw a frame."""
    if alpha <= 0.01:
        return
    W, H = surf.get_size()
    cx, cy = W // 2, int(H * 0.62)
    box = pygame.Rect(0, 0, 440, 150)
    box.center = (cx, cy)
    pygame.draw.rect(surf, T.with_alpha(T.PANEL, 0.72 * alpha), box, border_radius=8)
    pygame.draw.rect(surf, T.with_alpha(T.INK_FAINT, alpha), box, 1, border_radius=8)

    # animation timeline (3 s loop): 0-1 fingers close, 1-2 hands pull apart drawing a rect, 2-3 hold
    cycle = (t % 3.0) / 3.0
    close = min(1.0, cycle / 0.3)
    spread = 0.0 if cycle < 0.33 else min(1.0, (cycle - 0.33) / 0.4)
    ease = 1 - (1 - spread) ** 3
    demo_c = (box.x + 110, box.centery)
    dx, dy = 20 + 70 * ease, 12 + 34 * ease
    pa = (demo_c[0] - dx, demo_c[1] - dy)
    pb = (demo_c[0] + dx, demo_c[1] + dy)
    gap = 14 * (1 - close)
    for p, col in ((pa, T.VIOLET), (pb, T.CYAN)):
        pygame.draw.circle(surf, T.with_alpha(col, alpha), (p[0] - gap, p[1] - gap * 0.4), 5, 2)
        pygame.draw.circle(surf, T.with_alpha(col, alpha), (p[0] + gap, p[1] + gap * 0.4), 5, 2)
        if close >= 1:
            pygame.draw.circle(surf, T.with_alpha(col, 0.35 * alpha), p, 11)
    if spread > 0:
        rect = pygame.Rect(min(pa[0], pb[0]), min(pa[1], pb[1]), abs(pb[0] - pa[0]), abs(pb[1] - pa[1]))
        pygame.draw.rect(surf, T.with_alpha(T.GREEN, 0.9 * alpha * spread), rect, 1)

    x = box.x + 210
    lines = [  # (font, text, colour, space after)
        (T.font(13, bold=True), "Pinch to interact", T.INK, 8),
        (T.font(12), "Touch thumb and index finger.", T.INK_DIM, 3),
        (T.font(12), "Pinch with both hands and pull", T.INK_DIM, 3),
        (T.font(12), "apart to draw a frame.", T.INK_DIM, 10),
        (T.font(10, mono=True), "H · all gestures and shortcuts", T.INK_FAINT, 0),
    ]
    y = box.y + 20
    for f, text, col, gap in lines:
        s = f.render(text, True, col)
        s.set_alpha(int(255 * alpha))
        surf.blit(s, (x, y))
        y += f.get_height() + gap


HELP_GESTURES = [
    ("Pinch (thumb + index)", "Select a frame / grab"),
    ("Pinch + move", "Drag frame"),
    ("Quick pinch tap on a frame", "Zoom: open / close fingers"),
    ("Pinch on a corner", "Resize from that corner"),
    ("Two pinches, empty space", "Draw a new frame"),
    ("Two pinches on a frame", "Scale + rotate"),
    ("Hands apart / together", "Scale up / down"),
    ("Drag to ⊗ at the bottom", "Delete frame"),
    ("Quick pinch in empty space", "Clear selection"),
    ("Open hand", "Release interaction"),
    ("Open hand swipe ← →", "Previous / next effect"),
    ("Hold V-sign (index + middle)", "Switch mode"),
]

HELP_KEYS = [
    ("Tab  ·  hold V-sign", "Next mode"),
    ("1–9, 0  ·  ← →", "Choose effect (18)"),
    ("E", "Effects on / off"),
    ("C  ·  A", "Calibration  ·  auto-calibrate"),
    ("S", "Hand skeleton"),
    ("F  /  F11", "Fullscreen"),
    ("Del  ·  Z", "Delete / clear · undo"),
    ("G  ·  [ ]", "Gravity · brush size"),
    ("Mouse drag", "Draw / move frames"),
    ("P  ·  T", "Screenshot · theme neon/retro"),
    ("H", "This help"),
    ("Q  /  Esc", "Quit"),
]


def draw_help(surf: pygame.Surface) -> None:
    W, H = surf.get_size()
    box = pygame.Rect(0, 0, min(760, W - 40), min(460, H - 40))
    box.center = (W // 2, H // 2)
    pygame.draw.rect(surf, T.with_alpha(T.PANEL, 0.92), box, border_radius=8)
    pygame.draw.rect(surf, T.with_alpha(T.INK_FAINT, 1.0), box, 1, border_radius=8)
    pygame.draw.line(surf, T.CYAN, (box.x + 24, box.y), (box.x + 90, box.y), 2)
    surf.blit(spaced("GESTURES", 12, T.INK, 4, mono=True, bold=True), (box.x + 28, box.y + 24))
    surf.blit(spaced("KEYBOARD", 12, T.INK, 4, mono=True, bold=True), (box.x + box.width // 2 + 40, box.y + 24))
    f, fd = T.font(12, bold=True), T.font(12)
    y = box.y + 56
    for g, a in HELP_GESTURES:
        surf.blit(f.render(g, True, T.INK), (box.x + 28, y))
        surf.blit(fd.render(a, True, T.INK_DIM), (box.x + 28, y + 15))
        y += 29
        if y > box.bottom - 40:
            break
    y = box.y + 56
    fm = T.font(12, mono=True)
    for k, a in HELP_KEYS:
        surf.blit(fm.render(k, True, T.CYAN), (box.x + box.width // 2 + 40, y))
        surf.blit(fd.render(a, True, T.INK_DIM), (box.x + box.width // 2 + 200, y))
        y += 26
    hint = T.font(10, mono=True).render("PRESS H TO CLOSE", True, T.INK_FAINT)
    surf.blit(hint, hint.get_rect(midbottom=(box.centerx, box.bottom - 12)))
