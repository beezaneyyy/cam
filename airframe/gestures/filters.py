"""Signal smoothing.

The One Euro filter (Casiez et al., 2012) is an adaptive low-pass filter: at
low speeds it smooths aggressively (removing jitter), at high speeds it raises
its cutoff (removing lag). It is the standard choice for hand/pointer input.

This implementation is vectorised: it filters a scalar, a point (shape (2,))
or a whole landmark array (shape (21, 3)). The speed used to adapt the cutoff
is the Euclidean norm along the last axis, so each point is filtered
isotropically.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = None
        self._t = None

    @property
    def value(self) -> np.ndarray | None:
        return self._x

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x.copy()
        dt = t - self._t
        if dt <= 1e-6:
            return self._x.copy()
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        if self._dx.ndim == 0:
            speed = np.abs(self._dx)
        else:
            speed = np.linalg.norm(self._dx, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()


def exp_smoothing_factor(speed: float, dt: float) -> float:
    """Frame-rate independent factor for ``value += (target - value) * k``."""
    return 1.0 - math.exp(-speed * max(dt, 0.0))
