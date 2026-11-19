"""Interaction controller: maps gestures (and the mouse) onto frame manipulations.

Per-hand state machine
----------------------
A pinch that starts…

* **on a corner handle** of the selected frame → RESIZE (opposite corner fixed)
* **inside a frame**                           → DRAG (and select / bring to front)
* **while the other hand drags a frame** (and not inside a different frame)
                                               → TRANSFORM with both hands
                                                 (move + uniform scale + rotate)
* **in empty space**                           → ANCHOR; if the other hand is also
                                                 anchored, both become CREATE and a
                                                 rectangle spans the two pinch points.

Releasing:

* DRAG over the trash zone deletes the frame.
* TRANSFORM → the remaining hand continues as a DRAG.
* CREATE commits the rectangle (if large enough); the other hand is CONSUMED
  until it releases, so it cannot trigger anything by accident.
* A quick ANCHOR tap (short, no motion) clears the selection.

The mouse is fed through the same machine as a virtual hand that is allowed to
create a frame on its own by dragging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from airframe.config import InteractionConfig
from airframe.gestures.engine import EventType, GestureFrame

from .geometry import RectGeom, rect_from_points, resize_from_corner, two_point_transform
from .manager import FrameManager

MOUSE_ID = 0  # tracked hands use ids ≥ 1


class Mode(Enum):
    IDLE = "idle"
    ANCHOR = "anchor"
    CREATE = "create"
    DRAG = "drag"
    RESIZE = "resize"
    TRANSFORM = "transform"
    CONSUMED = "consumed"


@dataclass
class HandOp:
    mode: Mode
    start_pos: np.ndarray
    start_time: float
    frame_id: int | None = None
    offset: np.ndarray | None = None
    corner: int = -1
    geom0: RectGeom | None = None
    max_motion: float = 0.0
    single_create: bool = False


@dataclass
class TransformOp:
    frame_id: int
    hand_a: int
    hand_b: int
    geom0: RectGeom
    a0: np.ndarray
    b0: np.ndarray


@dataclass
class CreateOp:
    hand_a: int
    hand_b: int
    anchor: np.ndarray            # used when one pointer creates alone (mouse)
    preview: RectGeom | None = None


@dataclass
class ZoomOp:
    """Phone-style zoom armed on a frame by a tap; driven by one hand's thumb–index gap."""

    frame_id: int
    hand_id: int
    geom0: RectGeom
    armed_at: float
    gap0: float | None = None
    gap: float | None = None
    scale: float = 1.0
    ref_scale: float = 1.0
    still_since: float = 0.0
    changed: bool = False


@dataclass
class Notice:
    text: str
    time: float
    kind: str = "info"


@dataclass
class InteractionState:
    """What the renderer needs to know about ongoing interactions."""

    creation: RectGeom | None = None
    creation_valid: bool = False
    trash_active: bool = False
    trash_hover: bool = False
    trash_center: tuple[float, float] = (0.0, 0.0)
    modes: dict[int, Mode] = field(default_factory=dict)
    zoom_frame: int | None = None
    zoom_hand: int | None = None
    zoom_scale: float = 1.0
    zoom_lock: float = 0.0          # 0..1 progress toward locking the size
    zoom_active: bool = False       # fingers are driving the size (past the settle time)


class InteractionController:
    def __init__(
        self,
        manager: FrameManager,
        config: InteractionConfig,
        effect_names: list[str],
        default_effect: str | None = None,
        on_created=None,
    ):
        if not effect_names:
            raise ValueError("at least one effect is required")
        self.manager = manager
        self.config = config
        self.effect_names = list(effect_names)
        self.default_effect = default_effect if default_effect in self.effect_names else self.effect_names[0]
        self.on_created = on_created  # optional callback(frame) after a frame is committed
        self.ops: dict[int, HandOp] = {}
        self.transform: TransformOp | None = None
        self.creation: CreateOp | None = None
        self.zoom: ZoomOp | None = None
        self.hand_infos: dict = {}
        self.positions: dict[int, np.ndarray] = {}
        self.image_size: tuple[int, int] = (1280, 720)
        self.notices: list[Notice] = []
        self.state = InteractionState()
        self.now = 0.0

    # ------------------------------------------------------------------ public API
    def update(self, gf: GestureFrame | None, now: float) -> InteractionState:
        self.now = now
        if gf is not None:
            if gf.image_size[0]:
                self.image_size = gf.image_size
            self.hand_infos = {h.hand_id: h for h in gf.hands}
            for h in gf.hands:
                self.positions[h.hand_id] = np.asarray(h.interaction_point, dtype=np.float64)
            for ev in gf.events:
                pos = np.asarray(ev.position, dtype=np.float64)
                z = self.zoom
                # Note: OPEN_PALM does not end a zoom — spreading thumb and index with the other
                # fingers relaxed *is* an open hand, and that is exactly how people zoom in.
                if z is not None and ev.hand_id == z.hand_id and ev.type is EventType.HAND_LOST:
                    self._finish_zoom()
                if ev.type is EventType.PINCH_START:
                    self.positions[ev.hand_id] = pos
                    if z is not None and self.zoom is not None and ev.hand_id == z.hand_id:
                        # closing the fingers all the way while zooming = smallest size, not a drag
                        self.ops[ev.hand_id] = HandOp(Mode.CONSUMED, pos.copy(), now)
                        continue
                    if self.zoom is not None:  # the other hand starts something: confirm the zoom
                        self._finish_zoom()
                    self._begin(ev.hand_id, pos, now)
                elif ev.type in (EventType.PINCH_END, EventType.HAND_LOST, EventType.OPEN_PALM):
                    if ev.hand_id in self.ops:
                        if ev.type is EventType.PINCH_END:
                            self.positions[ev.hand_id] = pos
                        self._end(ev.hand_id, now, released=ev.type is EventType.PINCH_END)
                    if ev.type is EventType.HAND_LOST:
                        self.positions.pop(ev.hand_id, None)
                elif ev.type is EventType.SWIPE:
                    if ev.hand_id not in self.ops:
                        self.cycle_effect(ev.direction)
        self._tick()
        return self.state

    # Mouse / pointer -------------------------------------------------------------
    def pointer_down(self, pos, now: float, pointer_id: int = MOUSE_ID) -> None:
        p = np.asarray(pos, dtype=np.float64)
        self.positions[pointer_id] = p
        self._begin(pointer_id, p, now, single_create=True)
        self._tick()

    def pointer_move(self, pos, pointer_id: int = MOUSE_ID) -> None:
        self.positions[pointer_id] = np.asarray(pos, dtype=np.float64)

    def pointer_up(self, pos, now: float, pointer_id: int = MOUSE_ID) -> None:
        self.positions[pointer_id] = np.asarray(pos, dtype=np.float64)
        self._tick()
        self._end(pointer_id, now)
        self._tick()

    # Effects / commands ----------------------------------------------------------
    def current_effect(self) -> str:
        sel = self.manager.selected
        return sel.effect if sel is not None else self.default_effect

    def set_effect(self, name: str) -> None:
        if name not in self.effect_names:
            return
        sel = self.manager.selected
        if sel is not None:
            sel.effect = name
            sel.effect_changed_at = self.now
        self.default_effect = name
        self.notify(f"EFFECT · {name.upper().replace('_', ' ')}")

    def cycle_effect(self, direction: int) -> str:
        idx = self.effect_names.index(self.current_effect())
        name = self.effect_names[(idx + direction) % len(self.effect_names)]
        self.set_effect(name)
        return name

    def delete_selected(self) -> bool:
        sel = self.manager.selected
        if sel is None:
            return False
        self._forget_frame(sel.frame_id)
        self.manager.remove(sel.frame_id)
        self.notify("FRAME REMOVED", "warn")
        return True

    def clear_all(self) -> None:
        for f in self.manager.active_frames:
            self._forget_frame(f.frame_id)
        self.manager.clear()
        self.notify("ALL FRAMES CLEARED", "warn")

    # Zoom ----------------------------------------------------------------------------
    def _start_zoom(self, frame, hid: int, now: float) -> None:
        if self.zoom is not None:
            self._finish_zoom(quiet=True)
        self.zoom = ZoomOp(frame.frame_id, hid, frame.target, now, still_since=now)
        frame.held = True
        self.notify("ZOOM · OPEN / CLOSE FINGERS")

    def _finish_zoom(self, quiet: bool = False) -> None:
        z = self.zoom
        self.zoom = None
        if z is None:
            return
        frame = self.manager.get(z.frame_id)
        if frame is not None:
            frame.held = False
            if not quiet and z.changed:
                self.notify(f"SIZE LOCKED · {z.scale * 100:.0f}%")

    def scale_frame(self, frame_id: int, factor: float) -> None:
        """Uniformly scale a frame about its centre (mouse wheel)."""
        frame = self.manager.get(frame_id)
        if frame is None:
            return
        g = frame.target
        factor = max(factor, self.config.min_frame_size / max(min(g.w, g.h), 1e-6))
        frame.target = RectGeom(g.cx, g.cy, g.w * factor, g.h * factor, g.angle)

    def _tick_zoom(self, state: "InteractionState") -> None:
        z, cfg, now = self.zoom, self.config, self.now
        frame = self.manager.get(z.frame_id)
        if frame is None:
            self.zoom = None
            return
        h = self.hand_infos.get(z.hand_id)
        if h is not None and now - z.armed_at >= cfg.zoom_settle_seconds:
            r = float(h.pinch_ratio)
            z.gap = r if z.gap is None else z.gap + (r - z.gap) * 0.35
            if z.gap0 is None:
                z.gap0 = max(z.gap, 0.15)
                z.still_since = now
            s = (max(z.gap, 0.05) / z.gap0) ** cfg.zoom_exponent
            g0 = z.geom0
            s_min = cfg.min_frame_size / max(min(g0.w, g0.h), 1e-6)
            z.scale = float(np.clip(s, min(s_min, 1.0), cfg.zoom_max_scale))
            g = frame.target
            frame.target = RectGeom(g.cx, g.cy, g0.w * z.scale, g0.h * z.scale, g.angle)
            if abs(z.scale - z.ref_scale) > 0.03:
                z.ref_scale, z.still_since, z.changed = z.scale, now, True
        if z.changed:
            lock = (now - z.still_since) / cfg.zoom_lock_seconds
        else:
            lock = (now - z.armed_at) / cfg.zoom_idle_seconds
        state.zoom_frame, state.zoom_hand, state.zoom_scale = z.frame_id, z.hand_id, z.scale
        state.zoom_lock = float(min(1.0, lock))
        state.zoom_active = z.gap0 is not None
        if lock >= 1.0:
            self._finish_zoom()

    def cancel_all(self) -> None:
        """Drop every ongoing interaction (used when switching to another mode)."""
        for op in self.ops.values():
            frame = self.manager.get(op.frame_id)
            if frame is not None:
                frame.held = False
        self.ops.clear()
        self.transform = None
        self.creation = None
        self._finish_zoom(quiet=True)
        self.positions.clear()
        self._tick()

    def mode_of(self, hand_id: int) -> Mode:
        op = self.ops.get(hand_id)
        return op.mode if op else Mode.IDLE

    def trash_center(self) -> np.ndarray:
        w, h = self.image_size
        # bottom centre, kept clear of the effect bar drawn along the bottom edge
        return np.array([w / 2.0, h - self.config.trash_radius - 0.16 * h])

    # ------------------------------------------------------------------ internals
    def notify(self, text: str, kind: str = "info") -> None:
        self.notices.append(Notice(text, self.now, kind))
        del self.notices[:-6]

    def _forget_frame(self, frame_id: int) -> None:
        """Drop any operation that targets a frame about to disappear."""
        for hid, op in self.ops.items():
            if op.frame_id == frame_id:
                op.mode, op.frame_id = Mode.CONSUMED, None
        if self.transform is not None and self.transform.frame_id == frame_id:
            self.transform = None
        if self.zoom is not None and self.zoom.frame_id == frame_id:
            self.zoom = None

    def _begin(self, hid: int, p: np.ndarray, now: float, single_create: bool = False) -> None:
        if hid in self.ops:  # duplicate start: restart cleanly
            self._end(hid, now)
        mgr = self.manager
        hit = mgr.hit_test(p, margin=6.0)

        # 1. second hand joins a hand that is already holding a frame
        for oid, op in list(self.ops.items()):
            if op.mode in (Mode.DRAG, Mode.RESIZE) and op.frame_id is not None:
                if hit is None or hit.frame_id == op.frame_id:
                    self._start_transform(op.frame_id, oid, hid, p, now)
                    return

        # 2. corner handle of the selected frame
        corner = mgr.hit_corner(p, self.config.handle_radius, selected_only=True)
        if corner is not None and not self._frame_busy(corner[0].frame_id):
            frame, idx = corner
            self.ops[hid] = HandOp(Mode.RESIZE, p.copy(), now, frame.frame_id, corner=idx, geom0=frame.target)
            frame.held = True
            return

        # 3. inside a frame
        if hit is not None and not self._frame_busy(hit.frame_id):
            mgr.select(hit.frame_id)
            mgr.bring_to_front(hit.frame_id)
            hit.held = True
            self.ops[hid] = HandOp(Mode.DRAG, p.copy(), now, hit.frame_id, offset=hit.target.center - p)
            return

        # 4. empty space: pair with an anchored hand to create a frame
        if not single_create:
            for oid, op in self.ops.items():
                if op.mode is Mode.ANCHOR and not op.single_create:
                    op.mode = Mode.CREATE
                    self.ops[hid] = HandOp(Mode.CREATE, p.copy(), now)
                    self.creation = CreateOp(oid, hid, op.start_pos.copy())
                    return
        self.ops[hid] = HandOp(Mode.ANCHOR, p.copy(), now, single_create=single_create)

    def _frame_busy(self, frame_id: int) -> bool:
        return any(op.frame_id == frame_id for op in self.ops.values())

    def _start_transform(self, frame_id: int, holder: int, joiner: int, p: np.ndarray, now: float) -> None:
        frame = self.manager.get(frame_id)
        if frame is None:
            self.ops[joiner] = HandOp(Mode.CONSUMED, p.copy(), now)
            return
        a0 = self.positions.get(holder, self.ops[holder].start_pos).copy()
        self.transform = TransformOp(frame_id, holder, joiner, frame.target, a0, p.copy())
        self.ops[holder].mode = Mode.TRANSFORM
        self.ops[joiner] = HandOp(Mode.TRANSFORM, p.copy(), now, frame_id)
        frame.held = True
        self.manager.select(frame_id)

    def _end(self, hid: int, now: float, released: bool = False) -> None:
        """``released``: the pinch was genuinely opened (a lost hand or open palm is not a tap)."""
        op = self.ops.pop(hid, None)
        if op is None:
            return
        mgr = self.manager
        pos = self.positions.get(hid, op.start_pos)
        frame = mgr.get(op.frame_id)

        if op.mode is Mode.DRAG:
            if frame is not None:
                frame.held = False
                if self._in_trash(pos):
                    mgr.remove(frame.frame_id)
                    self.notify("FRAME REMOVED", "warn")
                elif (
                    self.config.zoom_enabled
                    and released
                    and hid != MOUSE_ID
                    and now - op.start_time <= self.config.tap_max_seconds
                    and op.max_motion <= self.config.tap_max_motion
                ):
                    self._start_zoom(frame, hid, now)
        elif op.mode is Mode.RESIZE:
            if frame is not None:
                frame.held = False
        elif op.mode is Mode.TRANSFORM:
            t = self.transform
            self.transform = None
            if t is not None:
                other = t.hand_b if hid == t.hand_a else t.hand_a
                other_op = self.ops.get(other)
                frame = mgr.get(t.frame_id)
                if other_op is not None and frame is not None:
                    other_pos = self.positions.get(other, other_op.start_pos)
                    other_op.mode = Mode.DRAG
                    other_op.frame_id = frame.frame_id
                    other_op.offset = frame.target.center - other_pos
                elif frame is not None:
                    frame.held = False
        elif op.mode is Mode.CREATE:
            c = self.creation
            self.creation = None
            if c is not None and c.preview is not None:
                created = mgr.create(c.preview, self.default_effect)
                if created is not None:
                    mgr.select(created.frame_id)
                    if self.on_created is not None:
                        self.on_created(created)
                    else:
                        self.notify(f"FRAME {created.frame_id:02d} · {created.effect.upper().replace('_', ' ')}")
            if c is not None:
                other = c.hand_b if hid == c.hand_a else c.hand_a
                if other != hid and other in self.ops:
                    self.ops[other].mode = Mode.CONSUMED
        elif op.mode is Mode.ANCHOR:
            if now - op.start_time <= self.config.tap_max_seconds and op.max_motion <= self.config.tap_max_motion:
                mgr.select(None)

    def _in_trash(self, p) -> bool:
        return float(np.linalg.norm(np.asarray(p) - self.trash_center())) <= self.config.trash_radius

    def _tick(self) -> None:
        mgr, cfg = self.manager, self.config
        state = InteractionState(trash_center=tuple(self.trash_center()))
        for hid, op in list(self.ops.items()):
            p = self.positions.get(hid)
            if p is None:
                continue
            op.max_motion = max(op.max_motion, float(np.linalg.norm(p - op.start_pos)))
            if op.mode is Mode.DRAG and op.frame_id is not None:
                frame = mgr.get(op.frame_id)
                if frame is not None and op.offset is not None:
                    mgr.set_target(frame.frame_id, frame.target.with_center(p + op.offset))
                    state.trash_active = True
                    if self._in_trash(p):
                        state.trash_hover = True
            elif op.mode is Mode.RESIZE and op.geom0 is not None and op.frame_id is not None:
                mgr.set_target(op.frame_id, resize_from_corner(op.geom0, op.corner, p, cfg.min_frame_size))
            elif op.mode is Mode.ANCHOR and op.single_create and op.max_motion > 12.0:
                op.mode = Mode.CREATE
                self.creation = CreateOp(hid, hid, op.start_pos.copy())

        t = self.transform
        if t is not None:
            pa, pb = self.positions.get(t.hand_a), self.positions.get(t.hand_b)
            if pa is not None and pb is not None:
                geom = two_point_transform(
                    t.geom0, t.a0, t.b0, pa, pb,
                    min_size=cfg.min_frame_size,
                    max_scale=cfg.max_scale_step,
                    snap=np.radians(cfg.rotation_snap_degrees),
                )
                mgr.set_target(t.frame_id, geom)

        c = self.creation
        if c is not None:
            a = self.positions.get(c.hand_a, c.anchor) if c.hand_a != c.hand_b else c.anchor
            b = self.positions.get(c.hand_b)
            if b is not None:
                c.preview = rect_from_points(a, b)
                state.creation = c.preview
                state.creation_valid = min(c.preview.w, c.preview.h) >= cfg.min_frame_size

        if self.zoom is not None:
            self._tick_zoom(state)
        state.modes = {hid: op.mode for hid, op in self.ops.items()}
        self.state = state
