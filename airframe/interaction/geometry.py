"""Rotated-rectangle geometry used by virtual frames.

A rectangle is described by its center, size and rotation angle (radians,
counter-clockwise in image coordinates is *not* assumed — the angle is applied
as a plain 2D rotation of the local axes, which renders clockwise on screen
because the image y axis points down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# Corner order in local space: top-left, top-right, bottom-right, bottom-left
_CORNER_SIGNS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)


@dataclass(frozen=True)
class RectGeom:
    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float64)

    def with_center(self, c) -> "RectGeom":
        return replace(self, cx=float(c[0]), cy=float(c[1]))


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate(v, angle: float) -> np.ndarray:
    return rotation_matrix(angle) @ np.asarray(v, dtype=np.float64)


def corners(g: RectGeom) -> np.ndarray:
    """(4, 2) world-space corners: TL, TR, BR, BL."""
    local = _CORNER_SIGNS * np.array([g.w / 2.0, g.h / 2.0])
    return local @ rotation_matrix(g.angle).T + g.center


def to_local(g: RectGeom, p) -> np.ndarray:
    return rotate(np.asarray(p, dtype=np.float64) - g.center, -g.angle)


def to_world(g: RectGeom, local) -> np.ndarray:
    return rotate(local, g.angle) + g.center


def contains(g: RectGeom, p, margin: float = 0.0) -> bool:
    lx, ly = to_local(g, p)
    return abs(lx) <= g.w / 2.0 + margin and abs(ly) <= g.h / 2.0 + margin


def nearest_corner(g: RectGeom, p) -> tuple[int, float]:
    pts = corners(g)
    d = np.linalg.norm(pts - np.asarray(p, dtype=np.float64), axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def rect_from_points(a, b) -> RectGeom:
    """Axis-aligned rectangle spanned by two opposite corners."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = (a + b) / 2.0
    w, h = np.abs(b - a)
    return RectGeom(float(c[0]), float(c[1]), float(w), float(h), 0.0)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def snap_angle(angle: float, snap: float) -> float:
    """Snap to the nearest multiple of 90° when within ``snap`` radians."""
    if snap <= 0:
        return angle
    quarter = math.pi / 2.0
    nearest = round(angle / quarter) * quarter
    return nearest if abs(angle - nearest) <= snap else angle


def two_point_transform(
    g0: RectGeom,
    a0,
    b0,
    a1,
    b1,
    min_size: float = 1.0,
    max_scale: float = 8.0,
    snap: float = 0.0,
) -> RectGeom:
    """Similarity transform defined by two tracked points (pinch-to-zoom/rotate).

    Points ``a0, b0`` were grabbed when the gesture began on rectangle ``g0``;
    ``a1, b1`` are their current positions. The rectangle follows the
    translation of the midpoint, the change of distance (uniform scale) and
    the change of angle (rotation) between the two points.
    """
    a0, b0, a1, b1 = (np.asarray(p, dtype=np.float64) for p in (a0, b0, a1, b1))
    v0, v1 = b0 - a0, b1 - a1
    d0, d1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if d0 < 1e-6 or d1 < 1e-6:
        return g0
    scale = float(np.clip(d1 / d0, 1.0 / max_scale, max_scale))
    min_dim = max(min(g0.w, g0.h), 1e-6)
    if min_dim * scale < min_size:  # never shrink below the minimum size
        scale = min_size / min_dim
    dangle = math.atan2(v1[1], v1[0]) - math.atan2(v0[1], v0[0])
    angle = wrap_angle(g0.angle + dangle)
    snapped = snap_angle(angle, snap)
    dangle += snapped - angle
    m0, m1 = (a0 + b0) / 2.0, (a1 + b1) / 2.0
    center = m1 + rotate((g0.center - m0) * scale, dangle)
    return RectGeom(float(center[0]), float(center[1]), g0.w * scale, g0.h * scale, snapped)


def resize_from_corner(g0: RectGeom, corner: int, p, min_size: float = 1.0) -> RectGeom:
    """Move one corner to ``p`` while the opposite corner stays fixed (rotation kept)."""
    opposite_local = -_CORNER_SIGNS[corner] * np.array([g0.w / 2.0, g0.h / 2.0])
    anchor = to_world(g0, opposite_local)
    local = rotate(np.asarray(p, dtype=np.float64) - anchor, -g0.angle)
    sign = _CORNER_SIGNS[corner]
    # extent along the direction of the dragged corner, clamped to min size
    ex = max(local[0] * sign[0], min_size)
    ey = max(local[1] * sign[1], min_size)
    center_local = np.array([ex * sign[0], ey * sign[1]]) / 2.0
    center = anchor + rotate(center_local, g0.angle)
    return RectGeom(float(center[0]), float(center[1]), float(ex), float(ey), g0.angle)


def lerp_geom(a: RectGeom, b: RectGeom, t: float) -> RectGeom:
    t = float(min(1.0, max(0.0, t)))
    da = wrap_angle(b.angle - a.angle)
    return RectGeom(
        a.cx + (b.cx - a.cx) * t,
        a.cy + (b.cy - a.cy) * t,
        a.w + (b.w - a.w) * t,
        a.h + (b.h - a.h) * t,
        wrap_angle(a.angle + da * t),
    )


def geom_distance(a: RectGeom, b: RectGeom) -> float:
    """Rough magnitude of the difference between two geometries (pixels)."""
    diag = math.hypot(a.w, a.h)
    return (
        abs(a.cx - b.cx) + abs(a.cy - b.cy) + abs(a.w - b.w) + abs(a.h - b.h)
        + abs(wrap_angle(a.angle - b.angle)) * diag * 0.5
    )
