import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from airframe.rendering import pixelfont
from airframe.rendering import theme as T


@pytest.fixture(autouse=True)
def _pygame():
    pygame.init()
    yield
    T.set_theme("neon")
    pygame.quit()
    T.reset_fonts()


def test_pixel_font_normalizes_accents_and_case():
    assert pixelfont.normalize("Não foi possível") == "NAO FOI POSSIVEL"
    assert pixelfont.normalize("3 × 4 − 1 → Σ") == "3 × 4 - 1 → Σ"
    assert pixelfont.normalize("✌") == "?"


def test_pixel_font_metrics_and_render():
    f = pixelfont.PixelFont(14)
    assert f.scale == 2 and pixelfont.PixelFont(10).scale == 1 and pixelfont.PixelFont(64).scale == 6
    w, h = f.size("AB")
    assert (w, h) == ((2 * 6 - 1) * 2, 9 * 2)
    s = f.render("AB", True, (10, 20, 30))
    assert s.get_size() == (w, h)
    arr = pygame.surfarray.pixels_alpha(s)
    assert set(np.unique(arr)) <= {0, 255}  # crisp: no anti-aliasing
    s.set_alpha(10)  # mutating a returned surface must not poison the cache
    assert f.render("AB", True, (10, 20, 30)).get_alpha() != 10


def test_theme_switch_recolours_and_swaps_fonts():
    T.set_theme("retro")
    assert T.is_retro() and T.PANEL == T.PALETTES["retro"]["PANEL"]
    assert isinstance(T.font(12), pixelfont.PixelFont)
    assert T.HAND_COLORS["Right"] == T.PALETTES["retro"]["CYAN"]
    T.set_theme("neon")
    assert not isinstance(T.font(12), pixelfont.PixelFont)
    assert T.set_theme("nope") == "neon"
