"""Stable hand identities across frames.

MediaPipe returns hands in arbitrary order. Interactions (a pinch that drags a
frame, a two-hand resize) need a stable id per physical hand, so detections are
matched to existing tracks by palm position, with a penalty when handedness
disagrees. With at most two hands, trying every permutation is trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .landmarks import RawHand


@dataclass
class _Track:
    track_id: int
    position: np.ndarray
    handedness: str
    missed: int = 0


class HandAssociator:
    def __init__(self, max_distance: float, max_missed: int = 5, handedness_penalty: float = 0.25):
        """``max_distance`` is in pixels; ``handedness_penalty`` is a fraction of it."""
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.handedness_penalty = handedness_penalty * max_distance
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()

    def _cost(self, track: _Track, hand: RawHand) -> float:
        cost = float(np.linalg.norm(track.position - hand.palm_center))
        if track.handedness != hand.handedness:
            cost += self.handedness_penalty
        return cost

    def assign(self, hands: list[RawHand]) -> list[RawHand]:
        """Set ``track_id`` on every hand in place and return them."""
        tracks = self._tracks
        best: list[tuple[int, int]] = []
        best_cost = float("inf")
        n_t, n_h = len(tracks), len(hands)
        k = min(n_t, n_h)
        if k:
            for t_idx in permutations(range(n_t), k):
                for h_idx in permutations(range(n_h), k):
                    pairs = list(zip(t_idx, h_idx))
                    costs = [self._cost(tracks[t], hands[h]) for t, h in pairs]
                    valid = [(t, h) for (t, h), c in zip(pairs, costs) if c <= self.max_distance]
                    # prefer more matches first, then lower total cost
                    total = sum(c for c in costs if c <= self.max_distance) - 1e6 * len(valid)
                    if total < best_cost:
                        best_cost = total
                        best = valid
        matched_tracks = set()
        matched_hands = set()
        for t, h in best:
            track = tracks[t]
            hand = hands[h]
            hand.track_id = track.track_id
            track.position = hand.palm_center
            track.handedness = hand.handedness
            track.missed = 0
            matched_tracks.add(t)
            matched_hands.add(h)
        for i, track in enumerate(tracks):
            if i not in matched_tracks:
                track.missed += 1
        for h, hand in enumerate(hands):
            if h not in matched_hands:
                hand.track_id = self._next_id
                self._next_id += 1
                tracks.append(_Track(hand.track_id, hand.palm_center, hand.handedness))
        self._tracks = [t for t in tracks if t.missed <= self.max_missed]
        return hands
