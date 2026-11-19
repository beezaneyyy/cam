"""Virtual frame model."""

from __future__ import annotations

from dataclasses import dataclass, field

from airframe.gestures.filters import exp_smoothing_factor

from .geometry import RectGeom, geom_distance, lerp_geom


@dataclass
class VirtualFrame:
    """A rectangle living in camera pixel space that carries a visual effect.

    ``target`` is where interactions want the frame to be; ``display`` eases
    toward it every tick. Rendering always uses ``display``, which is what keeps
    frames visually stable when hands jitter and smooth when they jump.
    """

    frame_id: int
    target: RectGeom
    effect: str
    display: RectGeom = field(default=None)  # type: ignore[assignment]
    opacity: float = 0.0       # appear / disappear animation, 0..1
    removing: bool = False
    age: float = 0.0
    held: bool = False         # currently manipulated by a hand
    effect_changed_at: float = -10.0

    def __post_init__(self) -> None:
        if self.display is None:
            # start slightly smaller so creation "pops" into place
            t = self.target
            self.display = RectGeom(t.cx, t.cy, t.w * 0.92, t.h * 0.92, t.angle)

    @property
    def alive(self) -> bool:
        return not (self.removing and self.opacity <= 0.01)

    @property
    def settled(self) -> bool:
        return geom_distance(self.display, self.target) < 0.5

    def update(self, dt: float, follow_speed: float) -> None:
        self.age += dt
        k = exp_smoothing_factor(follow_speed, dt)
        self.display = lerp_geom(self.display, self.target, k)
        goal = 0.0 if self.removing else 1.0
        self.opacity += (goal - self.opacity) * exp_smoothing_factor(12.0, dt)
        if self.removing and self.opacity < 0.02:
            self.opacity = 0.0
