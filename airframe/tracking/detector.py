"""MediaPipe Tasks hand landmarker wrapper."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from airframe.config import TrackingConfig

from .landmarks import NUM_LANDMARKS, RawHand

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class DetectorError(RuntimeError):
    pass


class HandDetector:
    """Runs the hand landmarker in VIDEO mode (uses temporal tracking between frames)."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        model = Path(config.model_path)
        if not model.is_file():
            raise DetectorError(
                f"Modelo de mãos não encontrado em {model}.\n"
                "Execute scripts/download_model.sh para baixá-lo."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:  # pragma: no cover - depends on install
            raise DetectorError(f"MediaPipe não está instalado: {exc}") from exc
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - surface any native init error
            raise DetectorError(f"Falha ao inicializar o detector de mãos: {exc}") from exc
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[RawHand]:
        """Detect hands. Returned landmarks are in ``frame_bgr`` pixel units."""
        h, w = frame_bgr.shape[:2]
        dw = min(self.config.detect_width, w)
        if dw < w:
            dh = int(round(h * dw / w))
            small = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # VIDEO mode demands strictly increasing timestamps
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return convert_result(result.hand_landmarks, result.handedness, w, h)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # noqa: BLE001
            pass


def convert_result(hand_landmarks, handedness, width: int, height: int) -> list[RawHand]:
    """Convert MediaPipe normalized landmarks into pixel-space ``RawHand`` objects."""
    hands: list[RawHand] = []
    for i, lms in enumerate(hand_landmarks):
        arr = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
        for j, lm in enumerate(lms[:NUM_LANDMARKS]):
            arr[j, 0] = lm.x * width
            arr[j, 1] = lm.y * height
            arr[j, 2] = lm.z * width
        label, score = "Unknown", 1.0
        if handedness and i < len(handedness) and handedness[i]:
            cat = handedness[i][0]
            label = cat.category_name or cat.display_name or "Unknown"
            score = float(cat.score)
        hands.append(RawHand(arr, label, score))
    return hands
