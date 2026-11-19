"""Interaction modes. FRAMES (virtual frames + effects) lives in airframe.interaction;
the others are self-contained ``Experience`` subclasses."""

from .base import Experience, Pointer, PointerTracker
from .count import CountMode
from .cutout import CutoutMode
from .draw import DrawMode
from .energy import EnergyMode
from .particles import ParticlesMode
from .physics import PhysicsMode
from .web import WebMode

# name → factory(settings). "frames" is not here: it is the built-in virtual-frame system.
MODE_FACTORIES = {
    "cutout": lambda s: CutoutMode(s.interaction),
    "web": lambda s: WebMode(),
    "count": lambda s: CountMode(),
    "draw": lambda s: DrawMode(),
    "physics": lambda s: PhysicsMode(),
    "energy": lambda s: EnergyMode(),
    "particles": lambda s: ParticlesMode(),
}

__all__ = [
    "MODE_FACTORIES",
    "CountMode",
    "CutoutMode",
    "DrawMode",
    "EnergyMode",
    "Experience",
    "ParticlesMode",
    "PhysicsMode",
    "Pointer",
    "PointerTracker",
    "WebMode",
]
