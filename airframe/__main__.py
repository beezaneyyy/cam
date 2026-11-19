"""Command-line entry point: ``python -m airframe``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Quiet native logging from MediaPipe/TFLite before anything imports it.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from airframe.config import settings as settings_mod  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="airframe", description="AIRFRAME — hand tracking visual experience")
    p.add_argument("--config", type=Path, default=settings_mod.DEFAULT_CONFIG_PATH, help="TOML config file")
    p.add_argument("--camera", help="camera index (e.g. 0) or path to a video file")
    p.add_argument("--resolution", help="capture resolution, e.g. 1280x720 or 640x480")
    p.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the camera image")
    p.add_argument("--fps", type=int, help="render frame-rate cap (default 60)")
    p.add_argument("--reset-calibration", action="store_true", help="ignore the saved calibration")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    calibration = None if args.reset_calibration else settings_mod.DEFAULT_CALIBRATION_PATH
    settings = settings_mod.load_settings(args.config, calibration)
    if args.camera is not None:
        settings.camera.device = int(args.camera) if args.camera.isdigit() else args.camera
    if args.resolution:
        try:
            w, h = (int(v) for v in args.resolution.lower().split("x"))
        except ValueError:
            print("--resolution must look like 1280x720", file=sys.stderr)
            return 2
        settings.camera.width, settings.camera.height = w, h
    if args.fullscreen:
        settings.ui.fullscreen = True
    if args.no_mirror:
        settings.camera.mirror = False
    if args.fps:
        settings.ui.target_fps = args.fps

    from airframe.app import App

    return App(settings).run()


if __name__ == "__main__":
    sys.exit(main())
