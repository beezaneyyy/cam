"""Hand landmark definitions shared by tracking and gesture modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

# (tip, pip) for the four long fingers
FINGERS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class RawHand:
    """One detected hand.

    ``landmarks`` is a (21, 3) float32 array in camera pixel units: x and y are
    image coordinates, z is MediaPipe's relative depth scaled by image width so
    all three axes share the same unit.
    """

    landmarks: np.ndarray
    handedness: str = "Unknown"   # "Left" / "Right" from the user's point of view
    score: float = 1.0
    track_id: int = -1

    @property
    def palm_center(self) -> np.ndarray:
        return self.landmarks[list(PALM), :2].mean(axis=0)


@dataclass
class TrackingResult:
    hands: list[RawHand] = field(default_factory=list)
    frame_seq: int = 0
    timestamp: float = 0.0
    image_size: tuple[int, int] = (0, 0)   # (width, height) of the camera frame
    inference_ms: float = 0.0
