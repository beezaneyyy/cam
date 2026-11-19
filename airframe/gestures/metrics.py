"""Geometric measurements on hand landmarks.

All functions take a (21, 3) landmark array in a single isotropic unit
(pixels for x/y, z scaled by image width), so ratios are invariant to the
distance between the hand and the camera.
"""

from __future__ import annotations

import numpy as np

from airframe.tracking.landmarks import (
    FINGERS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PALM,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
)


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def palm_size(lm: np.ndarray) -> float:
    """Per-hand length reference that survives hand rotation.

    Palm length (wrist → middle MCP) foreshortens when the fingers point at the
    camera; palm width (index MCP → pinky MCP) foreshortens when the hand turns
    sideways. They rarely shrink together, so the larger of the two (width
    scaled to match length on an average hand) stays stable.
    """
    length = distance(lm[WRIST], lm[MIDDLE_MCP])
    width = distance(lm[INDEX_MCP], lm[PINKY_MCP]) * 1.25
    return max(length, width, 1e-6)


def pinch_ratio(lm: np.ndarray) -> float:
    """Thumb-tip ↔ index-tip distance normalised by palm size (≈0.1 pinched, ≈1 open)."""
    return distance(lm[THUMB_TIP], lm[INDEX_TIP]) / palm_size(lm)


def pinch_point(lm: np.ndarray) -> np.ndarray:
    """2D midpoint between thumb tip and index tip."""
    return (lm[THUMB_TIP, :2] + lm[INDEX_TIP, :2]) * 0.5


def palm_center(lm: np.ndarray) -> np.ndarray:
    return lm[list(PALM), :2].mean(axis=0)


def finger_extended(lm: np.ndarray, tip: int, pip: int, ratio: float = 1.15) -> bool:
    """A finger is extended when its tip is clearly farther from the wrist than its PIP joint.

    Measuring against the wrist makes the test independent of hand rotation.
    """
    return distance(lm[WRIST], lm[tip]) > ratio * distance(lm[WRIST], lm[pip])


def extended_fingers(lm: np.ndarray, ratio: float = 1.15) -> int:
    return sum(finger_extended(lm, tip, pip, ratio) for tip, pip in FINGERS)


def thumb_extended(lm: np.ndarray, threshold: float = 0.55) -> bool:
    return distance(lm[THUMB_TIP], lm[INDEX_MCP]) / palm_size(lm) > threshold


def is_open_hand(lm: np.ndarray, ratio: float = 1.15, pinch_off: float = 0.42) -> bool:
    return (
        extended_fingers(lm, ratio) == 4
        and thumb_extended(lm)
        and pinch_ratio(lm) > pinch_off
    )


def thumb_up(lm: np.ndarray) -> bool:
    """Thumb extended away from the palm (for finger counting).

    Two conditions, both rotation invariant: the tip is well away from the
    index knuckle, and it reaches farther from the pinky knuckle than the
    thumb's own IP joint does (a folded thumb curls back across the palm).
    """
    from airframe.tracking.landmarks import THUMB_IP

    away = distance(lm[THUMB_TIP], lm[INDEX_MCP]) / palm_size(lm) > 0.6
    out = distance(lm[THUMB_TIP], lm[PINKY_MCP]) > 1.08 * distance(lm[THUMB_IP], lm[PINKY_MCP])
    return away and out


def finger_states(lm: np.ndarray, ratio: float = 1.15) -> list[bool]:
    """[thumb, index, middle, ring, pinky] extended?"""
    return [thumb_up(lm)] + [finger_extended(lm, tip, pip, ratio) for tip, pip in FINGERS]


def count_fingers(lm: np.ndarray, ratio: float = 1.15) -> int:
    return sum(finger_states(lm, ratio))


def is_ok_sign(lm: np.ndarray, ratio: float = 1.15, pinch_on: float = 0.22) -> bool:
    """👌 — thumb and index touching while middle, ring and pinky are extended."""
    (_, _), (mt, mp), (rt, rp), (pt, pp) = FINGERS
    return (
        pinch_ratio(lm) < pinch_on * 1.3
        and finger_extended(lm, mt, mp, ratio)
        and finger_extended(lm, rt, rp, ratio)
        and finger_extended(lm, pt, pp, ratio)
    )


def is_peace(lm: np.ndarray, ratio: float = 1.15) -> bool:
    """✌ — index and middle extended and spread apart, ring and pinky folded."""
    (it, ip), (mt, mp), (rt, rp), (pt, pp) = FINGERS
    if not (finger_extended(lm, it, ip, ratio) and finger_extended(lm, mt, mp, ratio)):
        return False
    if finger_extended(lm, rt, rp, 1.0) or finger_extended(lm, pt, pp, 1.0):
        return False
    spread = distance(lm[it], lm[mt]) / palm_size(lm)
    return spread > 0.25 and pinch_ratio(lm) > 0.45


def is_fist(lm: np.ndarray, ratio: float = 1.15) -> bool:
    return extended_fingers(lm, ratio) == 0
