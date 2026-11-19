"""End-to-end pipeline test without a webcam or the MediaPipe model.

A fake camera and a fake tracking worker feed synthetic frames and hands into
the real App (gestures → interaction → effects → pygame rendering), using
SDL's dummy video driver so it runs headless.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import pytest

from airframe.app import App
from airframe.camera import CameraStatus, FramePacket
from airframe.config import Settings
from airframe.rendering import theme as T
from airframe.tracking.landmarks import TrackingResult

from .helpers import raw_hand


class FakeCamera:
    def __init__(self, w=1280, h=720):
        y, x = np.mgrid[0:h, 0:w]
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = (x * 200 // w).astype(np.uint8)
        img[..., 1] = (y * 200 // h).astype(np.uint8)
        img[..., 2] = 90
        img.setflags(write=False)  # the app must never write into the shared frame
        self.packet = FramePacket(img, 1, 0.0)
        self.status = CameraStatus.RUNNING
        self.resolution = (w, h)
        self.fps = 30.0

    def latest(self):
        return self.packet

    def stop(self):
        pass


class FakeWorker:
    fps = 30.0
    inference_ms = 12.0

    def __init__(self):
        self.result = None
        self.seq = 0

    def push(self, hands, t):
        self.seq += 1
        self.result = TrackingResult(hands, self.seq, t, (1280, 720), 12.0)

    def latest(self):
        return self.seq, self.result

    def stop(self):
        pass


@pytest.fixture
def app():
    s = Settings()
    s.ui.window_width, s.ui.window_height = 1280, 720
    s.gestures.pinch_on_frames = 1
    s.gestures.pinch_off_frames = 1
    s.gestures.pinch_warmup_frames = 0
    a = App(s)
    pygame.init()
    a._open_window()
    a.clock = pygame.time.Clock()
    a._build_ui()
    a.camera = FakeCamera()
    a.worker = FakeWorker()
    a.boot_steps = [["x", "done"]]
    a.phase = "run"
    a.run_started = -10.0
    yield a
    pygame.quit()
    T.reset_fonts()


def step(app, t, left=None, right=None):
    hands = []
    if left is not None:
        hands.append(raw_hand(left[0], pinch=left[1], handedness="Left", track_id=1))
    if right is not None:
        hands.append(raw_hand(right[0], pinch=right[1], handedness="Right", track_id=2))
    app.worker.push(hands, t)
    app._run_frame(1 / 30, t)


def test_full_pipeline_creates_moves_and_renders_frame(app):
    t = 0.0
    # both hands appear open, then pinch → creation → release
    for _ in range(3):
        t += 1 / 30
        step(app, t, ((300, 450), False), ((900, 650), False))
    for _ in range(3):
        t += 1 / 30
        step(app, t, ((300, 450), True), ((900, 650), True))
    assert app.controller.state.creation is not None
    for _ in range(3):
        t += 1 / 30
        step(app, t, ((300, 450), False), ((900, 650), False))
    assert len(app.manager) == 1
    frame = app.manager.active_frames[0]
    assert frame.effect == "neon"
    assert frame.target.w > 400 and frame.target.h > 150

    # rendered pixels inside the frame differ from the plain graded background
    for _ in range(20):
        t += 1 / 30
        step(app, t)
    screen = pygame.surfarray.array3d(app.screen)
    assert screen.std() > 5

    # switch effect with the keyboard and render again
    app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_4, mod=0), t)
    assert frame.effect == "glitch"
    t += 1 / 30
    step(app, t)

    # toggle effects, calibration, help, skeleton: all must render without error
    for key in (pygame.K_e, pygame.K_c, pygame.K_h, pygame.K_s):
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0), t)
        t += 1 / 30
        step(app, t, ((500, 500), False))


def test_mouse_creates_frame_through_viewport(app):
    t = 1.0
    step(app, t)
    down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(400, 300))
    move = pygame.event.Event(pygame.MOUSEMOTION, pos=(700, 520), rel=(300, 220), buttons=(1, 0, 0))
    up = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(700, 520))
    for ev in (down, move):
        app._handle_event(ev, t)
        t += 0.05
        step(app, t)
    app._handle_event(up, t)
    assert len(app.manager) == 1


def test_screens_render():
    pygame.init()
    surf = pygame.display.set_mode((960, 540))
    from airframe.ui import screens

    screens.draw_splash(surf, 0.5, [("Camera", "done"), ("Model", "active"), ("Tracking", "pending")])
    screens.draw_error(surf, 0.5, "Camera unavailable", "Não foi possível abrir a câmera /dev/video0.")
    layer = pygame.Surface((960, 540), pygame.SRCALPHA)
    screens.draw_onboarding(layer, 1.2, 1.0)
    screens.draw_help(layer)
    pygame.quit()
    T.reset_fonts()


def test_every_mode_runs_and_renders_in_app(app):
    t = 0.0
    seen = []
    for _ in range(len(app.mode_names)):
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0), t)
        seen.append(app.mode)
        for i in range(12):  # both hands, pinching in the middle of the run
            t += 1 / 30
            pinch = 4 <= i < 9
            step(app, t, ((350 + i * 5, 450), pinch), ((850 - i * 5, 500), pinch))
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, mod=0), t)
    assert seen == ["cutout", "web", "count", "draw", "physics", "frames"]


def test_mouse_draws_in_draw_mode(app):
    app.set_mode(app.mode_names.index("draw"))
    t = 0.0
    app._handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(400, 300)), t)
    for x in range(400, 700, 30):
        t += 1 / 30
        app._handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(x, 300), rel=(30, 0), buttons=(1, 0, 0)), t)
        step(app, t)
    app._handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(700, 300)), t)
    step(app, t + 0.03)
    assert len(app.experience.strokes) == 1 and len(app.experience.strokes[0].points) > 5
