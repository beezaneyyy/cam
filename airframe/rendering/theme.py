"""Visual language: palettes and typography.

Two themes:
* ``neon``  — dark lab HUD, thin cyan/violet lines, modern fonts (default)
* ``retro`` — 2000s handheld-console menu: light rounded panels, blue borders,
              built-in 5×7 pixel font (interface only — the camera is not filtered)

Colours are module attributes read at draw time (``T.CYAN``), so switching the
theme at runtime with ``set_theme`` recolours everything immediately.
"""

from __future__ import annotations

from functools import lru_cache

import pygame

from . import pixelfont

PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    "neon": {
        "BG": (6, 8, 14),
        "INK": (226, 236, 248),
        "INK_DIM": (128, 142, 165),
        "INK_FAINT": (70, 82, 104),
        "CYAN": (40, 230, 255),
        "BLUE": (70, 130, 255),
        "VIOLET": (160, 100, 255),
        "MAGENTA": (255, 70, 200),
        "GREEN": (80, 255, 170),
        "AMBER": (255, 190, 70),
        "RED": (255, 80, 100),
        "PANEL": (10, 14, 24),
    },
    "retro": {
        "BG": (206, 214, 228),
        "INK": (36, 42, 62),
        "INK_DIM": (84, 94, 120),
        "INK_FAINT": (136, 146, 170),
        "CYAN": (24, 150, 232),
        "BLUE": (40, 96, 216),
        "VIOLET": (132, 76, 216),
        "MAGENTA": (222, 56, 150),
        "GREEN": (36, 168, 84),
        "AMBER": (236, 140, 16),
        "RED": (222, 48, 60),
        "PANEL": (246, 248, 252),
    },
}
THEMES = list(PALETTES)

# Defaults (neon) so the names exist at import time
BG = INK = INK_DIM = INK_FAINT = CYAN = BLUE = VIOLET = MAGENTA = GREEN = AMBER = RED = PANEL = (0, 0, 0)
HAND_COLORS: dict[str, tuple[int, int, int]] = {}
current = "retro"


def set_theme(name: str) -> str:
    global current
    if name not in PALETTES:
        name = "neon"
    current = name
    globals().update(PALETTES[name])
    HAND_COLORS.clear()
    HAND_COLORS.update({"Left": PALETTES[name]["VIOLET"], "Right": PALETTES[name]["CYAN"], "Unknown": PALETTES[name]["BLUE"]})
    font.cache_clear()
    return name


def is_retro() -> bool:
    return current == "retro"


_SANS = ["inter", "adwaitasans", "cantarell", "notosans", "dejavusans", "liberationsans"]
_MONO = ["jetbrainsmono", "jetbrainsmononl", "adwaitamono", "notosansmono", "dejavusansmono", "liberationmono"]


def _match(names: list[str], bold: bool = False) -> str | None:
    for name in names:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return path
    return None


@lru_cache(maxsize=64)
def font(size: int, mono: bool = False, bold: bool = False):
    if current == "retro":
        return pixelfont.PixelFont(size, bold)
    path = _match(_MONO if mono else _SANS, bold)
    f = pygame.font.Font(path, size)  # path=None → pygame's bundled default font
    if bold and path is None:
        f.set_bold(True)
    return f


def with_alpha(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    return (color[0], color[1], color[2], int(max(0, min(255, alpha * 255))))


def mix(a, b, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def reset_fonts() -> None:
    """Drop cached fonts (required after ``pygame.quit()``, which invalidates them)."""
    font.cache_clear()
    pixelfont.clear_cache()


set_theme("retro")
