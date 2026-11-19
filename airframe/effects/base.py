"""Effect interface and registry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EffectContext:
    time: float = 0.0       # seconds, drives animated effects
    seed: int = 0           # per-frame seed so two frames don't glitch in sync
    intensity: float = 1.0  # 0..1, used for transitions


class Effect:
    """Base class for region effects.

    ``apply`` receives a BGR uint8 region of interest (a *view* into the frame;
    it must not be modified) and returns a new array of the same shape and dtype.

    ``decorate`` may draw outside the region (e.g. glow) directly onto the full
    frame. ``polygon`` is the (4, 2) region outline in frame pixels.
    """

    name: str = "base"
    label: str = "Base"
    accent: tuple[int, int, int] = (255, 230, 80)  # RGB used by the UI for this effect

    def apply(self, roi: np.ndarray, ctx: EffectContext) -> np.ndarray:
        return roi.copy()

    def decorate(self, frame: np.ndarray, polygon: np.ndarray, ctx: EffectContext) -> None:
        return None


class EffectRegistry:
    def __init__(self) -> None:
        self._effects: dict[str, Effect] = {}

    def register(self, effect: Effect) -> Effect:
        if effect.name in self._effects:
            raise ValueError(f"duplicate effect {effect.name!r}")
        self._effects[effect.name] = effect
        return effect

    def get(self, name: str) -> Effect:
        return self._effects[name]

    def __contains__(self, name: str) -> bool:
        return name in self._effects

    @property
    def names(self) -> list[str]:
        return list(self._effects)

    def __iter__(self):
        return iter(self._effects.values())

    def __len__(self) -> int:
        return len(self._effects)
