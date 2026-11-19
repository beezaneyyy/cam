"""AIRFRAME application: wires camera → tracking → gestures → interaction → rendering.

Threads
-------
* camera thread   – grabs frames, keeps only the newest (airframe.camera)
* tracking thread – MediaPipe inference on the newest frame (airframe.tracking.worker)
* main thread     – gestures, interaction, effects, pygame rendering and input

The main thread never waits for inference: it renders the newest camera frame
with the newest tracking result available, so a slow detector lowers tracking
rate but never the UI frame rate.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pygame

from airframe.camera import CameraError, CameraStatus, CameraStream
from airframe.config import Settings, save_calibration
from airframe.effects import default_registry
from airframe.gestures import EventType, GestureEngine
from airframe.modes import MODE_FACTORIES, PointerTracker
from airframe.rendering.glow import GlowLayer
from airframe.interaction import FrameManager, InteractionController
from airframe.rendering import OverlayRenderer, SceneComposer, Viewport
from airframe.rendering import theme as T
from airframe.ui import Button, CalibrationPanel, HandBadge, Hud, HudData
from airframe.ui import screens

SPLASH_MIN_SECONDS = 1.6
SPLASH_FADE_SECONDS = 0.6


class App:
    def __init__(self, settings: Settings):
        self.s = settings
        self.registry = default_registry()
        self.manager = FrameManager(settings.interaction.max_frames, settings.interaction.min_frame_size)
        self.controller = InteractionController(
            self.manager, settings.interaction, self.registry.names, settings.ui.default_effect
        )
        self.gestures = GestureEngine(settings.gestures)
        self.composer = SceneComposer(self.registry)
        self.apply_theme(settings.ui.theme)
        self.overlay = OverlayRenderer(self.registry)
        self.viewport = Viewport()

        # Interaction modes. "frames" is the virtual-frame system above; the rest are Experiences.
        wanted = [m for m in settings.ui.modes if m == "frames" or m in MODE_FACTORIES] or ["frames"]
        self.experiences = {m: MODE_FACTORIES[m](settings) for m in wanted if m != "frames"}
        self.mode_names = wanted
        self.mode_labels = {"frames": ("Frames", T.CYAN), **{n: (e.label, e.accent) for n, e in self.experiences.items()}}
        self.mode_index = 0
        self.pointers = PointerTracker()
        self.glow = GlowLayer()
        self.mouse_cam: object = None

        self.camera: CameraStream | None = None
        self.detector = None
        self.worker = None

        self.phase = "boot"           # boot → run | error
        self.boot_steps: list[list[str]] = []
        self.boot_ok = False
        self.boot_started = 0.0
        self.error_title = ""
        self.error_message = ""

        self.effects_enabled = True
        self.show_skeleton = settings.ui.show_skeleton
        self.show_help = False
        self.fullscreen = settings.ui.fullscreen
        self.window_size = (settings.ui.window_width, settings.ui.window_height)
        self.run_started = 0.0
        self.onboarding_dismissed_at: float | None = None if settings.ui.show_onboarding else 0.0
        self.mouse_down = False
        self.running = True
        self._layer: pygame.Surface | None = None
        self._last_surface = None

    # ------------------------------------------------------------------ boot
    def _start_boot(self) -> None:
        self.boot_steps = [
            ["Requesting camera access", "pending"],
            ["Loading hand landmark model", "pending"],
            ["Starting real-time tracking", "pending"],
        ]
        self.boot_ok = False
        self.phase = "boot"
        self.boot_started = time.monotonic()
        threading.Thread(target=self._boot, name="boot", daemon=True).start()

    def _boot(self) -> None:
        from airframe.tracking.detector import DetectorError, HandDetector
        from airframe.tracking.worker import TrackingWorker

        steps = self.boot_steps
        try:
            steps[0][1] = "active"
            if self.camera is None:
                self.camera = CameraStream(self.s.camera)
                self.camera.start()
            steps[0][1] = "done"

            steps[1][1] = "active"
            if self.detector is None:
                self.detector = HandDetector(self.s.tracking)
            steps[1][1] = "done"

            steps[2][1] = "active"
            if self.worker is None:
                self.worker = TrackingWorker(self.camera, self.detector, self.s.tracking, self.s.gestures.lost_frames)
                self.worker.start()
            steps[2][1] = "done"
            self.boot_ok = True
        except CameraError as exc:
            steps[0][1] = "error"
            self.camera = None
            self._fail("Camera unavailable", str(exc))
        except DetectorError as exc:
            steps[1][1] = "error"
            self._fail("Hand tracking unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001 - show anything unexpected instead of crashing
            self._fail("Startup failed", f"{type(exc).__name__}: {exc}")

    def _fail(self, title: str, message: str) -> None:
        self.error_title, self.error_message = title, message
        self.phase = "error"

    # ------------------------------------------------------------------ window
    def _open_window(self) -> None:
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)

    def toggle_fullscreen(self) -> None:
        if not self.fullscreen:
            self.window_size = self.screen.get_size()
        self.fullscreen = not self.fullscreen
        self._open_window()

    def _get_layer(self) -> pygame.Surface:
        size = self.screen.get_size()
        if self._layer is None or self._layer.get_size() != size:
            self._layer = pygame.Surface(size, pygame.SRCALPHA)
        self._layer.fill((0, 0, 0, 0))
        return self._layer

    # ------------------------------------------------------------------ UI wiring
    def _build_ui(self) -> None:
        def toggle(attr):
            def _t():
                setattr(self, attr, not getattr(self, attr))
            return _t

        self.buttons = [
            Button("FX  ON/OFF", toggle("effects_enabled"), "E", is_active=lambda: self.effects_enabled),
            Button("CALIBRATE", self._toggle_calibration, "C", is_active=lambda: self.calibration.visible),
            Button("SKELETON", toggle("show_skeleton"), "S", is_active=lambda: self.show_skeleton),
            Button("FULLSCREEN", self.toggle_fullscreen, "F", is_active=lambda: self.fullscreen),
            Button("HELP", toggle("show_help"), "H", is_active=lambda: self.show_help),
        ]
        self.hud = Hud(self.registry, self.controller.set_effect, self.buttons, on_select_mode=self.set_mode)
        self.calibration = CalibrationPanel(
            self.s.gestures, self._on_calibration_change, self._save_calibration, self.controller.notify
        )

    @property
    def mode(self) -> str:
        return self.mode_names[self.mode_index]

    @property
    def experience(self):
        return self.experiences.get(self.mode)

    def set_mode(self, index: int) -> None:
        index %= len(self.mode_names)
        if index == self.mode_index:
            return
        if self.mode == "frames":
            self.controller.cancel_all()
        elif self.experience is not None:
            self.experience.exit()
        self.mode_index = index
        self.mouse_down = False
        if self.experience is not None:
            self.experience.enter()
        self.controller.notify(f"MODE · {self.mode_labels[self.mode][0].upper()}")

    def apply_theme(self, name: str) -> str:
        name = T.set_theme(name)
        self.s.ui.theme = name
        # The theme only changes the interface; the camera image stays untouched.
        return name

    def _screenshot(self) -> None:
        folder = Path(self.s.ui.screenshot_dir).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / time.strftime("airframe-%Y%m%d-%H%M%S.png")
            pygame.image.save(self.screen, str(path))
            self.controller.notify("SCREENSHOT SAVED")
            print(f"[airframe] screenshot: {path}")
        except (OSError, pygame.error) as exc:
            self.controller.notify(f"SCREENSHOT FAILED: {exc}", "warn")

    def _toggle_calibration(self) -> None:
        self.calibration.visible = not self.calibration.visible

    def _on_calibration_change(self) -> None:
        self.gestures.reconfigure(self.s.gestures)

    def _save_calibration(self) -> None:
        try:
            path = save_calibration(self.s)
            self.controller.notify("CALIBRATION SAVED")
            print(f"[airframe] calibration saved to {path}")
        except OSError as exc:
            self.controller.notify(f"SAVE FAILED: {exc}", "warn")

    # ------------------------------------------------------------------ input
    def _handle_event(self, ev: pygame.event.Event, now: float) -> None:
        if ev.type == pygame.QUIT:
            self.running = False
            return
        if self.phase == "error":
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                elif ev.key == pygame.K_r:
                    self._start_boot()
            return
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
            self.running = False
            return
        if self.phase != "run":
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self.running = False
            return

        if self.calibration.handle(ev):
            return
        if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION) and not self.mouse_down:
            if self.hud.handle(ev):
                return
        if ev.type == pygame.KEYDOWN:
            self._handle_key(ev, now)
        elif self.mode != "frames":
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self.mouse_down = True
                self.mouse_cam = self.viewport.to_camera(ev.pos)
            elif ev.type == pygame.MOUSEMOTION and self.mouse_down:
                self.mouse_cam = self.viewport.to_camera(ev.pos)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.mouse_down = False
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            pos = self.viewport.to_camera(ev.pos)
            if ev.button == 1:
                self.mouse_down = True
                self.controller.pointer_down(pos, now)
            elif ev.button == 3:
                hit = self.manager.hit_test(pos)
                if hit is not None:
                    self.manager.select(hit.frame_id)
                self.controller.cycle_effect(1)
            elif ev.button in (4, 5):
                hit = self.manager.hit_test(pos)
                if hit is not None:  # wheel over a frame = zoom it, like a phone pinch
                    self.controller.scale_frame(hit.frame_id, 1.1 if ev.button == 4 else 1 / 1.1)
                else:
                    self.controller.cycle_effect(1 if ev.button == 5 else -1)
        elif ev.type == pygame.MOUSEMOTION and self.mouse_down:
            self.controller.pointer_move(self.viewport.to_camera(ev.pos))
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.mouse_down:
            self.mouse_down = False
            self.controller.pointer_up(self.viewport.to_camera(ev.pos), now)

    def _handle_key(self, ev: pygame.event.Event, now: float) -> None:
        k = ev.key
        names = self.registry.names
        if k == pygame.K_ESCAPE:
            if self.show_help:
                self.show_help = False
            elif self.calibration.visible:
                self.calibration.visible = False
            elif self.fullscreen:
                self.toggle_fullscreen()
            else:
                self.running = False
        elif k in (pygame.K_f, pygame.K_F11):
            self.toggle_fullscreen()
        elif k == pygame.K_e:
            self.effects_enabled = not self.effects_enabled
            self.controller.notify("EFFECTS ON" if self.effects_enabled else "EFFECTS OFF")
        elif k == pygame.K_c:
            self._toggle_calibration()
        elif k == pygame.K_a:
            self.calibration.start_auto()
        elif k == pygame.K_s:
            self.show_skeleton = not self.show_skeleton
        elif k in (pygame.K_h, pygame.K_F1):
            self.show_help = not self.show_help
        elif k == pygame.K_p:
            self._screenshot()
        elif k == pygame.K_t:
            name = self.apply_theme("neon" if T.is_retro() else "retro")
            self.controller.notify(f"THEME · {name.upper()}")
        elif k == pygame.K_TAB:
            self.set_mode(self.mode_index + (-1 if ev.mod & pygame.KMOD_SHIFT else 1))
        elif self.experience is not None:
            exp = self.experience
            msg = exp.handle_key(k, ev.mod)
            if msg is None:
                if k in (pygame.K_DELETE, pygame.K_BACKSPACE, pygame.K_x):
                    msg = exp.clear()
                elif k == pygame.K_z:
                    msg = exp.undo()
            if msg:
                self.controller.notify(msg)
        elif k in (pygame.K_RIGHT, pygame.K_LEFT):
            self.controller.cycle_effect(1 if k == pygame.K_RIGHT else -1)
        elif pygame.K_1 <= k <= pygame.K_9 and k - pygame.K_1 < len(names):
            self.controller.set_effect(names[k - pygame.K_1])
        elif k == pygame.K_0 and len(names) >= 10:
            self.controller.set_effect(names[9])
        elif k in (pygame.K_DELETE, pygame.K_BACKSPACE, pygame.K_x):
            if ev.mod & pygame.KMOD_SHIFT:
                self.controller.clear_all()
            else:
                self.controller.delete_selected()

    # ------------------------------------------------------------------ frame
    def _run_frame(self, dt: float, now: float) -> None:
        cam, worker = self.camera, self.worker
        packet = cam.latest() if cam else None
        _, result = worker.latest() if worker else (0, None)
        gf = self.gestures.update(result, now)
        self.calibration.tick(now, [h.pinch_ratio for h in gf.hands if not h.stale])
        for ev in gf.events:
            if ev.type is EventType.PEACE and getattr(self.experience, "allows_mode_gesture", True):
                self.set_mode(self.mode_index + 1)
        exp = self.experience
        cam_size = packet.image.shape[1::-1] if packet is not None else self.controller.image_size
        if exp is None:
            state = self.controller.update(gf, now)
        else:
            state = self.controller.state
            mouse = (self.mouse_cam, True) if self.mouse_down and self.mouse_cam is not None else None
            if packet is not None:
                exp.on_camera(packet.image)
            exp.on_gestures(gf, now)
            exp.update(self.pointers.update(gf.hands, now, mouse), dt, now, tuple(cam_size))
            for ev in gf.events:
                if ev.type is EventType.SWIPE and hasattr(exp, "on_swipe"):
                    self.controller.notify(exp.on_swipe(ev.direction))
        self.manager.update(dt, self.s.interaction.follow_speed)
        if self.onboarding_dismissed_at is None and (len(self.manager) > 0 or self.mode != "frames"):
            self.onboarding_dismissed_at = now

        screen = self.screen
        W, H = screen.get_size()
        screen.fill(T.BG)
        if packet is not None:
            ph, pw = packet.image.shape[:2]
            self.viewport.update((pw, ph), (W, H))
            post = None
            if exp is not None:
                def post(image, exp=exp):
                    exp.render_image(image, now)
                    self.glow.begin((image.shape[1], image.shape[0]))
                    exp.render_light(self.glow, now)
                    self.glow.composite(image)
            frames = self.manager.frames if exp is None else []
            img = self.composer.compose(packet.image, frames, self.effects_enabled, now, self.viewport.size, post)
            self._last_surface = pygame.image.frombuffer(img.data, (img.shape[1], img.shape[0]), "BGR")
            self._last_img = img  # keep the buffer alive while the surface is used
        if self._last_surface is not None:
            x, y, _, _ = self.viewport.rect
            screen.blit(self._last_surface, (x, y))

        layer = self._get_layer()
        if cam is not None and cam.status is not CameraStatus.RUNNING:
            layer.fill((*T.BG, 170))
            msg = T.font(14, mono=True).render("CAMERA SIGNAL LOST — RECONNECTING…", True, T.AMBER)
            layer.blit(msg, msg.get_rect(center=(W // 2, H // 2)))

        view_rect = pygame.Rect(self.viewport.rect)
        if exp is None:
            self.overlay.draw(
                layer,
                self.viewport,
                self.manager.frames,
                self.manager.selected_id,
                gf.hands,
                state,
                self.s.interaction.trash_radius,
                self.s.interaction.handle_radius,
                self.show_skeleton,
                now,
            )
        else:
            self.overlay.draw_hands(layer, self.viewport, gf.hands, self.show_skeleton, now)
            exp.render_overlay(layer, self.viewport, now)
        sel = self.manager.selected
        data = HudData(
            camera_status=cam.status if cam else CameraStatus.STOPPED,
            camera_resolution=cam.resolution if cam else (0, 0),
            camera_fps=cam.fps if cam else 0.0,
            render_fps=self.clock.get_fps(),
            tracking_fps=worker.fps if worker else 0.0,
            inference_ms=worker.inference_ms if worker else 0.0,
            hands=[HandBadge(h.handedness, h.pinching, state.modes.get(h.hand_id), h.stale) for h in gf.hands],
            current_effect=self.controller.current_effect(),
            effect_target=f"FRAME {sel.frame_id:02d}" if sel else "NEW FRAMES",
            effects_enabled=self.effects_enabled,
            frame_count=len(self.manager),
            notices=sorted(self.controller.notices + list(getattr(exp, "notices", [])), key=lambda n: n.time),
            now=now,
            modes=[(n, *self.mode_labels[n]) for n in self.mode_names],
            mode_index=self.mode_index,
            hints=exp.hints if exp is not None else [],
        )
        self.hud.draw(layer, data, view_rect)
        self.calibration.draw(layer, [(h.handedness, h.pinch_ratio, h.pinching) for h in gf.hands])
        self.calibration.draw_auto(layer)
        screens.draw_onboarding(layer, now, self._onboarding_alpha(now))
        if self.show_help:
            screens.draw_help(layer)
        screen.blit(layer, (0, 0))

        since = now - self.run_started
        if since < SPLASH_FADE_SECONDS:
            screens.draw_splash(screen, now, [tuple(s) for s in self.boot_steps], 1.0 - since / SPLASH_FADE_SECONDS)

    def _onboarding_alpha(self, now: float) -> float:
        start = self.run_started + SPLASH_FADE_SECONDS
        fade_in = min(1.0, max(0.0, (now - start) / 0.6))
        end = start + self.s.ui.onboarding_seconds
        if self.onboarding_dismissed_at is not None:
            end = min(end, self.onboarding_dismissed_at + 0.4)
        fade_out = min(1.0, max(0.0, (end - now) / 0.6))
        return fade_in * fade_out

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        pygame.init()
        pygame.display.set_caption("AIRFRAME")
        self._open_window()
        self.clock = pygame.time.Clock()
        self._build_ui()
        self._start_boot()
        try:
            while self.running:
                dt = min(self.clock.tick(self.s.ui.target_fps) / 1000.0, 0.1)
                now = time.monotonic()
                for ev in pygame.event.get():
                    self._handle_event(ev, now)
                if self.phase == "boot":
                    screens.draw_splash(self.screen, now, [tuple(s) for s in self.boot_steps])
                    if self.boot_ok and now - self.boot_started >= SPLASH_MIN_SECONDS:
                        self.phase = "run"
                        self.run_started = now
                elif self.phase == "error":
                    screens.draw_error(self.screen, now, self.error_title, self.error_message)
                else:
                    self._run_frame(dt, now)
                pygame.display.flip()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.camera is not None:
            self.camera.stop()
        if self.detector is not None:
            self.detector.close()
        pygame.quit()
        T.reset_fonts()
