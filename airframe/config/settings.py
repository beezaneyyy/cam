"""Typed application settings.

Settings are plain dataclasses with sensible defaults. They can be overridden
by a TOML file (``airframe.toml``) and, for the gesture tolerances adjusted in
the calibration panel, by a per-user JSON file saved under
``~/.config/airframe/calibration.json``.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "airframe.toml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


def _user_config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "airframe"


DEFAULT_CALIBRATION_PATH = _user_config_dir() / "calibration.json"


@dataclass
class CameraConfig:
    device: int | str = 0          # V4L2 index or a video file path
    width: int = 1280
    height: int = 720
    fps: int = 30
    mjpg: bool = True              # MJPG is required by most UVC cams for 720p@30
    mirror: bool = True            # selfie view
    reconnect_seconds: float = 2.0


@dataclass
class TrackingConfig:
    model_path: str = str(DEFAULT_MODEL_PATH)
    num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    detect_width: int = 640        # frames are downscaled to this width before inference
    max_match_distance: float = 0.35  # fraction of image width for track association


@dataclass
class GestureConfig:
    # Pinch ratio = thumb-index distance / palm length. Hysteresis avoids flicker.
    pinch_on: float = 0.22
    pinch_off: float = 0.36
    pinch_on_frames: int = 3       # consecutive frames under pinch_on to engage
    pinch_off_frames: int = 3      # consecutive frames over pinch_off to release
    pinch_warmup_frames: int = 5   # a newly detected hand can't pinch until its landmarks settle
    # One Euro filter (pixel units)
    filter_min_cutoff: float = 1.2
    filter_beta: float = 0.015
    filter_d_cutoff: float = 1.0
    # Hand kept alive (holding its state) for this many missed tracking results
    lost_frames: int = 5
    # Open palm
    open_finger_ratio: float = 1.15  # tip-to-wrist / pip-to-wrist to count as extended
    # Horizontal swipe (open palm) to navigate effects
    swipe_enabled: bool = True
    swipe_min_distance: float = 0.28   # fraction of image width
    swipe_max_duration: float = 0.45   # seconds
    swipe_max_vertical_ratio: float = 0.5
    swipe_cooldown: float = 0.9


@dataclass
class InteractionConfig:
    min_frame_size: float = 60.0      # px (camera space)
    handle_radius: float = 34.0       # px, corner grab tolerance
    follow_speed: float = 22.0        # display smoothing (1/s), higher = snappier
    rotation_snap_degrees: float = 4.0
    max_scale_step: float = 4.0       # clamp of per-gesture two-hand scaling
    max_frames: int = 12
    trash_radius: float = 70.0        # px (camera space)
    tap_max_seconds: float = 0.35
    tap_max_motion: float = 25.0
    # Phone-style zoom: tap a frame, then open / close thumb and index to resize it
    zoom_enabled: bool = True
    zoom_exponent: float = 1.5        # >1 = more size change per finger movement
    zoom_settle_seconds: float = 0.25 # ignore the fingers right after the tap
    zoom_lock_seconds: float = 1.0    # hand still this long after zooming → size locked
    zoom_idle_seconds: float = 3.5    # tapped but never zoomed → disarm
    zoom_max_scale: float = 6.0


@dataclass
class UIConfig:
    window_width: int = 1280
    window_height: int = 720
    fullscreen: bool = False
    target_fps: int = 60
    show_skeleton: bool = True
    show_onboarding: bool = True
    onboarding_seconds: float = 9.0
    default_effect: str = "neon"
    # Modes in Tab / V-sign order. Available: frames, cutout, web, count, draw, physics, energy, particles
    modes: list[str] = field(default_factory=lambda: ["frames", "cutout", "web", "count", "draw", "physics"])
    screenshot_dir: str = "~/Pictures/AIRFRAME"
    theme: str = "neon"   # "neon" or "retro" (handheld-console look) — T toggles at runtime


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(obj: Any, data: dict[str, Any], strict: bool = False) -> None:
    """Recursively copy known keys from ``data`` into a dataclass instance."""
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            if strict:
                raise KeyError(f"Unknown setting: {key}")
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, strict)
        else:
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
            setattr(obj, key, value)


def load_settings(
    config_path: Path | str | None = DEFAULT_CONFIG_PATH,
    calibration_path: Path | str | None = DEFAULT_CALIBRATION_PATH,
) -> Settings:
    settings = Settings()
    if config_path is not None and Path(config_path).is_file():
        with open(config_path, "rb") as fh:
            _apply(settings, tomllib.load(fh))
    if calibration_path is not None and Path(calibration_path).is_file():
        try:
            data = json.loads(Path(calibration_path).read_text())
            _apply(settings.gestures, data.get("gestures", {}))
        except (OSError, ValueError):
            pass  # a corrupt calibration file must never prevent startup
    model = Path(settings.tracking.model_path).expanduser()
    if not model.is_absolute():
        model = PROJECT_ROOT / model
    settings.tracking.model_path = str(model)
    return settings


CALIBRATION_KEYS = (
    "pinch_on",
    "pinch_off",
    "filter_min_cutoff",
    "filter_beta",
    "swipe_min_distance",
    "swipe_enabled",
)


def save_calibration(settings: Settings, path: Path | str = DEFAULT_CALIBRATION_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"gestures": {k: getattr(settings.gestures, k) for k in CALIBRATION_KEYS}}
    path.write_text(json.dumps(data, indent=2))
    return path
