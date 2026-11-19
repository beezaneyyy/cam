"""Collection of virtual frames: creation, removal, selection, z-order, hit tests."""

from __future__ import annotations

from .frame import VirtualFrame
from .geometry import RectGeom, contains, nearest_corner


class FrameManager:
    def __init__(self, max_frames: int = 12, min_size: float = 40.0):
        self.max_frames = max_frames
        self.min_size = min_size
        self._frames: list[VirtualFrame] = []   # back → front
        self._next_id = 1
        self.selected_id: int | None = None

    # -- queries --------------------------------------------------------------------
    @property
    def frames(self) -> list[VirtualFrame]:
        """All frames including those fading out, back to front."""
        return list(self._frames)

    @property
    def active_frames(self) -> list[VirtualFrame]:
        return [f for f in self._frames if not f.removing]

    def __len__(self) -> int:
        return len(self.active_frames)

    def get(self, frame_id: int | None) -> VirtualFrame | None:
        if frame_id is None:
            return None
        for f in self._frames:
            if f.frame_id == frame_id and not f.removing:
                return f
        return None

    @property
    def selected(self) -> VirtualFrame | None:
        return self.get(self.selected_id)

    def hit_test(self, p, margin: float = 0.0) -> VirtualFrame | None:
        """Top-most frame containing ``p``."""
        for f in reversed(self._frames):
            if not f.removing and contains(f.target, p, margin):
                return f
        return None

    def hit_corner(self, p, radius: float, selected_only: bool = True) -> tuple[VirtualFrame, int] | None:
        candidates = [self.selected] if selected_only else list(reversed(self.active_frames))
        best: tuple[VirtualFrame, int] | None = None
        best_d = radius
        for f in candidates:
            if f is None:
                continue
            idx, d = nearest_corner(f.target, p)
            if d <= best_d:
                best, best_d = (f, idx), d
        return best

    # -- mutations ------------------------------------------------------------------
    def create(self, geom: RectGeom, effect: str) -> VirtualFrame | None:
        if geom.w < self.min_size or geom.h < self.min_size:
            return None
        if len(self.active_frames) >= self.max_frames:
            oldest = self.active_frames[0]
            self.remove(oldest.frame_id)
        frame = VirtualFrame(self._next_id, geom, effect)
        self._next_id += 1
        self._frames.append(frame)
        return frame

    def remove(self, frame_id: int, animate: bool = True) -> bool:
        frame = self.get(frame_id)
        if frame is None:
            return False
        if self.selected_id == frame_id:
            self.selected_id = None
        if animate:
            frame.removing = True
            frame.held = False
        else:
            self._frames.remove(frame)
        return True

    def clear(self) -> None:
        for f in self.active_frames:
            self.remove(f.frame_id)

    def select(self, frame_id: int | None) -> None:
        self.selected_id = frame_id if self.get(frame_id) is not None else None

    def bring_to_front(self, frame_id: int) -> None:
        frame = self.get(frame_id)
        if frame is not None:
            self._frames.remove(frame)
            self._frames.append(frame)

    def set_target(self, frame_id: int, geom: RectGeom) -> None:
        frame = self.get(frame_id)
        if frame is not None:
            frame.target = geom

    def update(self, dt: float, follow_speed: float) -> None:
        for f in self._frames:
            f.update(dt, follow_speed)
        self._frames = [f for f in self._frames if f.alive]
