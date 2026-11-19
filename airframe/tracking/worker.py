"""Background tracking thread.

Consumes the newest camera frame, runs the detector and publishes the newest
``TrackingResult``. Because it always skips to the latest frame, tracking never
falls behind the camera even when inference is slower than capture.
"""

from __future__ import annotations

import threading
import time

from airframe.camera import CameraStream
from airframe.config import TrackingConfig

from .association import HandAssociator
from .detector import HandDetector
from .landmarks import TrackingResult


class TrackingWorker:
    def __init__(self, camera: CameraStream, detector: HandDetector, config: TrackingConfig, lost_frames: int = 5):
        self.camera = camera
        self.detector = detector
        self.config = config
        self._lost_frames = lost_frames
        self._associator: HandAssociator | None = None
        self._lock = threading.Lock()
        self._result: TrackingResult | None = None
        self._result_id = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.fps = 0.0
        self.inference_ms = 0.0
        self.error: str | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        last_seq = 0
        t0 = time.monotonic()
        count = 0
        window = time.monotonic()
        while self._running:
            packet = self.camera.wait_newer(last_seq, timeout=0.25)
            if packet is None:
                continue
            last_seq = packet.seq
            h, w = packet.image.shape[:2]
            if self._associator is None:
                self._associator = HandAssociator(self.config.max_match_distance * w, self._lost_frames)
            start = time.perf_counter()
            try:
                hands = self.detector.detect(packet.image, int((packet.timestamp - t0) * 1000))
            except Exception as exc:  # noqa: BLE001 - keep the app alive
                self.error = str(exc)
                time.sleep(0.05)
                continue
            elapsed = (time.perf_counter() - start) * 1000
            self._associator.assign(hands)
            self.inference_ms = 0.9 * self.inference_ms + 0.1 * elapsed if self.inference_ms else elapsed
            result = TrackingResult(hands, packet.seq, packet.timestamp, (w, h), elapsed)
            with self._lock:
                self._result = result
                self._result_id += 1
            count += 1
            now = time.monotonic()
            if now - window >= 1.0:
                self.fps = count / (now - window)
                count = 0
                window = now

    def latest(self) -> tuple[int, TrackingResult | None]:
        """Return (result_id, result). ``result_id`` increases with each new result."""
        with self._lock:
            return self._result_id, self._result
