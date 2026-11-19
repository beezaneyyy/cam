"""Threaded webcam capture.

A background thread reads frames as fast as the device delivers them and keeps
only the most recent one, so slow consumers never build up latency. Frames are
mirrored once here (selfie view) and handed out as immutable snapshots: the
consumer receives the array the thread produced and the thread never touches it
again, so no extra copy is needed.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from airframe.config import CameraConfig


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened."""


class CameraStatus(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray   # BGR uint8, HxWx3 — treat as read-only
    seq: int            # monotonically increasing frame counter
    timestamp: float    # time.monotonic() at capture


def describe_open_failure(device: int | str) -> str:
    if isinstance(device, int):
        path = f"/dev/video{device}"
        return (
            f"Não foi possível abrir a câmera {path}.\n"
            "Verifique se a webcam está conectada, se outro programa não a está usando "
            "e se o seu usuário pertence ao grupo 'video'."
        )
    return f"Não foi possível abrir o vídeo '{device}'."


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest: FramePacket | None = None
        self._seq = 0
        self._running = False
        self.status = CameraStatus.STOPPED
        self.error: str | None = None
        self.resolution: tuple[int, int] = (0, 0)
        self.fps = 0.0
        self._is_file = isinstance(config.device, str) and not str(config.device).isdigit()

    # -- lifecycle -----------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture:
        dev = self.config.device
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        if isinstance(dev, int):
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(dev)  # fall back to any backend
        else:
            cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            raise CameraError(describe_open_failure(self.config.device))
        if not self._is_file:
            if self.config.mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            # Note: do NOT set CAP_PROP_BUFFERSIZE=1 — on many UVC drivers it halves
            # the frame rate. The capture thread already drops stale frames.
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise CameraError(
                "A câmera abriu, mas não entregou imagens. "
                "Ela pode estar em uso por outro aplicativo."
            )
        self.resolution = (frame.shape[1], frame.shape[0])
        return cap

    def start(self) -> None:
        """Open the device (blocking) and start the capture thread."""
        self.status = CameraStatus.STARTING
        try:
            self._cap = self._open()
        except CameraError as exc:
            self.status = CameraStatus.ERROR
            self.error = str(exc)
            raise
        self.error = None
        self._running = True
        self.status = CameraStatus.RUNNING
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status = CameraStatus.STOPPED

    # -- capture loop ----------------------------------------------------------------
    def _loop(self) -> None:
        frames = 0
        window_start = time.monotonic()
        file_period = 1.0 / max(1, self.config.fps)
        last_ok = time.monotonic()
        while self._running:
            cap = self._cap
            ok, frame = (False, None) if cap is None else cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                if self._is_file and cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
                    continue
                if now - last_ok > self.config.reconnect_seconds:
                    self._reconnect()
                    last_ok = time.monotonic()
                else:
                    time.sleep(0.01)
                continue
            last_ok = now
            if self.status is not CameraStatus.RUNNING:
                self.status = CameraStatus.RUNNING
            if self.config.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._seq += 1
                self._latest = FramePacket(frame, self._seq, now)
                self._new_frame.notify_all()
            frames += 1
            if now - window_start >= 1.0:
                self.fps = frames / (now - window_start)
                frames = 0
                window_start = now
            if self._is_file:
                time.sleep(file_period)

    def _reconnect(self) -> None:
        self.status = CameraStatus.RECONNECTING
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        while self._running:
            try:
                self._cap = self._open()
                self.status = CameraStatus.RUNNING
                return
            except CameraError as exc:
                self.error = str(exc)
                time.sleep(self.config.reconnect_seconds)

    # -- consumers ------------------------------------------------------------------
    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    def wait_newer(self, seq: int, timeout: float = 0.5) -> FramePacket | None:
        """Block until a frame newer than ``seq`` is available."""
        with self._new_frame:
            if self._latest is None or self._latest.seq <= seq:
                self._new_frame.wait(timeout)
            if self._latest is not None and self._latest.seq > seq:
                return self._latest
            return None
