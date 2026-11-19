import numpy as np
import pytest

from airframe.gestures import metrics
from airframe.gestures.engine import EventType, GestureEngine
from airframe.config import GestureConfig
from airframe.modes import DrawMode, EnergyMode, ParticlesMode, PhysicsMode, Pointer, PointerTracker
from airframe.modes.energy import lightning
from airframe.modes.particles import ParticleSystem
from airframe.rendering.glow import GlowLayer

from .helpers import hand_info, make_landmarks, raw_hand, tracking_result

SIZE = (1280, 720)
DT = 1 / 30


def ptr(pid, pos, pinching=False, is_open=False, vel=(0, 0), started=False, ended=False, palm_size=80.0):
    p = Pointer(pid, np.asarray(pos, float), pinching, is_open, palm=np.asarray(pos, float), palm_size=palm_size)
    p.velocity = np.asarray(vel, float)
    p.started, p.ended = started, ended
    return p


# ---------------------------------------------------------------- pointer tracker
def test_tracker_flags_pinch_edges_and_velocity():
    tr = PointerTracker(velocity_smoothing=0.0)
    a = tr.update([hand_info(1, (100, 100), pinching=False)], 0.0)[0]
    assert not a.started
    b = tr.update([hand_info(1, (130, 100), pinching=True)], 0.1)[0]
    assert b.started and b.velocity[0] == pytest.approx(300)
    c = tr.update([hand_info(1, (130, 100), pinching=False)], 0.2)[0]
    assert c.ended


def test_tracker_ends_pinch_of_vanished_pointer():
    tr = PointerTracker()
    tr.update([hand_info(1, (100, 100), pinching=True)], 0.0)
    out = tr.update([], 0.1)
    assert len(out) == 1 and out[0].ended and out[0].vanished


def test_tracker_mouse_pointer():
    tr = PointerTracker()
    p = tr.update([], 0.0, mouse=((10, 20), True))[0]
    assert p.pid == 0 and p.pinching and p.started


# ---------------------------------------------------------------- peace gesture
def _peace_landmarks():
    lm = make_landmarks()
    for tip, pip in ((16, 14), (20, 18)):     # fold ring + pinky
        lm[tip] = (lm[pip] - lm[0]) * 0.7 + lm[0]
    lm[8, 0] -= 25                             # spread index / middle apart
    lm[12, 0] += 15
    return lm


def test_is_peace():
    assert metrics.is_peace(_peace_landmarks())
    assert not metrics.is_peace(make_landmarks())          # open hand
    assert not metrics.is_peace(make_landmarks(fist=True))


def test_peace_held_fires_once():
    eng = GestureEngine(GestureConfig(pinch_warmup_frames=0))
    events = []
    for seq in range(1, 60):  # ~2 s
        h = raw_hand()
        h.landmarks = _peace_landmarks()
        gf = eng.update(tracking_result([h], seq, seq / 30), 0)
        events += [e.type for e in gf.events]
    assert events.count(EventType.PEACE) == 1


# ---------------------------------------------------------------- draw
def test_draw_pinch_creates_stroke_and_open_hand_erases():
    m = DrawMode()
    m.update([ptr(1, (100, 100), True, started=True)], DT, 0, SIZE)
    for x in range(110, 400, 10):
        m.update([ptr(1, (x, 100), True)], DT, 0, SIZE)
    m.update([ptr(1, (400, 100), False, ended=True)], DT, 0, SIZE)
    assert len(m.strokes) == 1 and len(m.strokes[0].points) > 20
    # wipe the middle with an open hand → stroke split in two
    m.update([ptr(2, (250, 100), is_open=True, palm_size=40)], DT, 0, SIZE)
    assert len(m.strokes) == 2
    xs = [p[0] for s in m.strokes for p in s.points]
    assert not any(230 < x < 270 for x in xs)


def test_draw_two_hands_draw_separate_strokes_and_undo_clear():
    m = DrawMode()
    m.update([ptr(1, (100, 100), True, started=True), ptr(2, (600, 100), True, started=True)], DT, 0, SIZE)
    m.update([ptr(1, (150, 100), True), ptr(2, (650, 100), True)], DT, 0, SIZE)
    m.update([ptr(1, (150, 100), ended=True), ptr(2, (650, 100), ended=True)], DT, 0, SIZE)
    assert len(m.strokes) == 2
    assert m.undo() and len(m.strokes) == 1
    m.clear()
    assert m.strokes == []


def test_draw_color_cycles_and_renders():
    m = DrawMode()
    assert "MAGENTA" in m.on_swipe(1)
    m.update([ptr(1, (100, 100), True, started=True)], DT, 0, SIZE)
    m.update([ptr(1, (300, 300), True)], DT, 0, SIZE)
    g = GlowLayer()
    g.begin(SIZE)
    m.render_light(g, 0.1)
    img = np.zeros((720, 1280, 3), np.uint8)
    g.composite(img)
    assert img[200, 200].max() > 50


# ---------------------------------------------------------------- particles
def test_particles_emit_attract_and_explode():
    ps = ParticleSystem(seed=1)
    ps.emit([(500, 300)], 200, spread=10)
    center = np.array([700, 300])
    d0 = np.linalg.norm(ps.pos - center, axis=1).mean()
    for _ in range(30):
        ps.attract(center, 900, DT)
        ps.step(DT, SIZE)
    assert np.linalg.norm(ps.pos - center, axis=1).mean() < d0
    before = np.linalg.norm(ps.vel, axis=1).mean()
    assert ps.impulse(center, 900, 400) > 0
    assert np.linalg.norm(ps.vel, axis=1).mean() > before


def test_particles_capacity_and_lifetime():
    ps = ParticleSystem(capacity=100)
    ps.emit([(10, 10)], 500, life=0.1)
    assert len(ps) == 100
    for _ in range(10):
        ps.step(DT, SIZE)
    assert len(ps) == 0


def test_particles_mode_pinch_release_creates_shockwave():
    m = ParticlesMode()
    for _ in range(10):
        m.update([ptr(1, (400, 300), vel=(500, 0))], DT, 0, SIZE)
    assert len(m.ps) > 0
    m.update([ptr(1, (400, 300), True, started=True)], DT, 0, SIZE)
    m.update([ptr(1, (400, 300), False, ended=True)], DT, 0.1, SIZE)
    assert len(m.shockwaves) == 1
    g = GlowLayer()
    g.begin(SIZE)
    m.render_light(g, 0.15)


# ---------------------------------------------------------------- energy
def test_lightning_endpoints_and_segments():
    rng = np.random.default_rng(0)
    pts = lightning((0, 0), (100, 0), rng, depth=4)
    assert len(pts) == 17
    np.testing.assert_allclose(pts[0], (0, 0))
    np.testing.assert_allclose(pts[-1], (100, 0))


def _hand_ptr(pid, pos, pinching):
    h = hand_info(pid, pos, pinching)
    h.landmarks = make_landmarks(center=pos)
    h.palm_center = np.asarray(pos, float)
    p = ptr(pid, pos, pinching)
    p.hand = h
    p.palm = np.asarray(pos, float)
    return p


def test_energy_orb_charge_and_blast():
    m = EnergyMode()
    t = 0.0
    m.update([_hand_ptr(1, (400, 400), False), _hand_ptr(2, (800, 400), False)], DT, t, SIZE)
    assert m.orb is not None
    np.testing.assert_allclose(m.orb[0], (600, 400))
    r_small = m.orb[1]
    for _ in range(45):
        t += DT
        m.update([_hand_ptr(1, (400, 400), True), _hand_ptr(2, (800, 400), True)], DT, t, SIZE)
    assert m.charge == pytest.approx(1.0)
    assert m.orb[1] > r_small
    m.update([_hand_ptr(1, (400, 400), False), _hand_ptr(2, (800, 400), False)], DT, t + DT, SIZE)
    assert len(m.blasts) == 1 and m.charge == 0
    g = GlowLayer()
    g.begin(SIZE)
    m.render_light(g, t + 0.1)
    assert g.dirty


# ---------------------------------------------------------------- physics
def test_physics_inflate_throw_and_gravity():
    m = PhysicsMode()
    m.update([ptr(1, (400, 300), True, started=True)], DT, 0, SIZE)
    for _ in range(20):
        m.update([ptr(1, (400, 300), True)], DT, 0, SIZE)
    ball = m.balls[0]
    assert ball.radius > 25 and ball.grabbed_by == 1
    m.update([ptr(1, (400, 300), False, vel=(800, -200), ended=True)], DT, 0, SIZE)
    assert ball.grabbed_by is None and ball.vel[0] > 500
    x0, vy0 = ball.pos[0], ball.vel[1]
    m.update([], DT, 0, SIZE)
    assert ball.pos[0] > x0 and ball.vel[1] > vy0  # moves and falls


def test_physics_grab_existing_ball_and_walls():
    m = PhysicsMode()
    m.gravity = False
    m.update([ptr(1, (100, 100), True, started=True)], DT, 0, SIZE)
    m.update([ptr(1, (100, 100), ended=True)], DT, 0, SIZE)
    b = m.balls[0]
    b.vel = np.array([-2000.0, 0.0])
    for _ in range(10):
        m.update([], DT, 0, SIZE)
    assert b.pos[0] >= b.radius and b.vel[0] > 0  # bounced off the left wall
    m.update([ptr(2, b.pos + 5, True, started=True)], DT, 0, SIZE)
    assert b.grabbed_by == 2 and len(m.balls) == 1  # grabbed, not a new ball


def test_physics_ball_collision_and_paddle():
    m = PhysicsMode()
    m.gravity = False
    from airframe.modes.physics import Ball

    a = Ball(np.array([300.0, 300.0]), np.array([400.0, 0.0]), 30, (255, 0, 0))
    b = Ball(np.array([355.0, 300.0]), np.array([0.0, 0.0]), 30, (0, 255, 0))
    m.balls = [a, b]
    m.update([], DT, 0, SIZE)
    assert b.vel[0] > 0 and a.vel[0] < 400
    c = Ball(np.array([700.0, 300.0]), np.array([0.0, 0.0]), 30, (0, 0, 255))
    m.balls = [c]
    m.update([ptr(3, (650, 300), is_open=True, vel=(600, 0), palm_size=60)], DT, 0, SIZE)
    assert c.vel[0] > 0 and c.pos[0] > 700


def test_glow_points_and_composite():
    g = GlowLayer()
    g.begin((200, 100))
    g.points(np.array([[50.0, 50.0], [-10, 5], [500, 500]]), np.array([[255, 255, 255]] * 3, np.uint8))
    img = np.zeros((100, 200, 3), np.uint8)
    g.composite(img)
    assert img[50, 50].max() > 100 and img[5, 190].max() == 0


# ---------------------------------------------------------------- cutout
from airframe.config import InteractionConfig  # noqa: E402
from airframe.gestures.engine import GestureEvent, GestureFrame  # noqa: E402
from airframe.interaction import RectGeom  # noqa: E402
from airframe.modes import CutoutMode, WebMode  # noqa: E402
from airframe.modes.cutout import CAPTURE_DELAY, warp_sticker  # noqa: E402
from airframe.modes.web import N, make_thread, simulate  # noqa: E402

from .helpers import GestureScript  # noqa: E402


def _camera():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[200:400, 300:600] = (0, 0, 255)  # red block = what we will cut
    return img


def _cut(m: CutoutMode, s: GestureScript, a=(300, 200), b=(600, 400), t=0.0):
    m.on_camera(_camera())
    for evs, dt_ in (([("start", 1, a)], 0), ([("start", 2, b)], 0.05), ([], 0.1), ([("end", 1)], 0.15), ([("end", 2)], 0.2)):
        m.on_gestures(s.frame(evs), t + dt_)
        m.update([], DT, t + dt_, SIZE)
    return t + 0.2


def test_cutout_captures_region_after_delay():
    m, s = CutoutMode(InteractionConfig(rotation_snap_degrees=0)), GestureScript()
    t = _cut(m, s)
    assert len(m.manager) == 1 and not m.stickers  # waiting for hands to move away
    m.on_camera(np.zeros((720, 1280, 3), np.uint8))  # hands gone, scene changed…
    m.update([], DT, t + CAPTURE_DELAY - 0.3, SIZE)
    assert not m.stickers
    m.on_camera(_camera())
    m.update([], DT, t + CAPTURE_DELAY + 0.05, SIZE)
    (st,) = m.stickers.values()
    assert st.image.shape == (200, 300, 3)
    assert (st.image[..., 2] == 255).all()


def test_cutout_sticker_moves_and_renders_where_dragged():
    m, s = CutoutMode(InteractionConfig(rotation_snap_degrees=0)), GestureScript()
    t = _cut(m, s)
    m.update([], DT, t + 2, SIZE)
    f = m.manager.active_frames[0]
    # drag the sticker 400 px to the right
    m.on_gestures(s.frame([("start", 1, (450, 300))]), t + 2.1)
    m.update([], DT, t + 2.1, SIZE)
    m.on_gestures(s.frame(h1=(850, 300)), t + 2.2)
    m.update([], DT, t + 2.2, SIZE)
    m.on_gestures(s.frame([("end", 1)]), t + 2.3)
    for i in range(60):
        m.update([], DT, t + 2.3 + i * DT, SIZE)
    assert f.display.cx == pytest.approx(850, abs=1)
    out = np.zeros((720, 1280, 3), np.uint8)
    m.render_image(out, t + 5)
    assert out[300, 850, 2] > 200   # sticker pixels at the new place
    assert out[300, 450].max() == 0  # nothing left at the old place (background is live, not drawn here)


def test_cutout_duplicate_and_clear():
    m, s = CutoutMode(), GestureScript()
    t = _cut(m, s)
    m.update([], DT, t + 2, SIZE)
    assert m.duplicate_selected() == "CLONED"
    m.update([], DT, t + 2.1, SIZE)
    assert len(m.stickers) == 2
    m.clear()
    for i in range(40):
        m.update([], DT, t + 3 + i * DT, SIZE)
    assert not m.stickers


def test_warp_sticker_scales_and_rotates():
    img = np.zeros((400, 400, 3), np.uint8)
    sticker = np.full((50, 100, 3), 200, np.uint8)
    warp_sticker(img, sticker, RectGeom(200, 200, 200, 100, np.pi / 2))
    assert img[200, 200].max() == 200
    assert img[110, 200].max() == 200 and img[200, 110].max() == 0  # now tall, not wide


# ---------------------------------------------------------------- web
def _two_hands(pid_a, pa, pid_b, pb, palm=90.0):
    """Pointers whose fingertips are all at the given points (enough for thread logic)."""
    out = []
    for pid, pos in ((pid_a, pa), (pid_b, pb)):
        p = _hand_ptr(pid, pos, False)
        p.palm_size = palm
        side = -1 if pid == pid_a else 1  # other fingertips spread away from each other: only index tips meet
        p.hand.landmarks[[4, 8, 12, 16, 20], :2] = np.array(pos) + np.array(
            [[side * 200, 100], [0, 0], [side * 250, 60], [side * 300, 90], [side * 350, 120]]
        )
        out.append(p)
    return out


def test_web_touch_forms_thread_and_slow_pull_stretches():
    m = WebMode()
    t = 0.0
    m.update(_two_hands(1, (600, 300), 2, (605, 300)), DT, t, SIZE)  # index tips touch
    threads = [th for th in m.threads if th.dying_since is None]
    assert len(threads) == 1 and {threads[0].a, threads[0].b} == {(1, 1), (2, 1)}
    for i in range(1, 90):  # pull apart slowly: 2 px/frame each = 120 px/s total
        t += DT
        m.update(_two_hands(1, (600 - 2 * i, 300), 2, (605 + 2 * i, 300)), DT, t, SIZE)
    th = threads[0]
    assert th.dying_since is None and th.rest > 250


def test_web_yank_snaps_thread():
    m = WebMode()
    t = 0.0
    m.update(_two_hands(1, (600, 300), 2, (605, 300)), DT, t, SIZE)
    snapped = False
    for i in range(1, 15):  # 40 px/frame each = 2400 px/s
        t += DT
        m.update(_two_hands(1, (600 - 40 * i, 300), 2, (605 + 40 * i, 300)), DT, t, SIZE)
        snapped = snapped or any(th.dying_since is not None for th in m.threads)
    assert snapped and m.sparks


def test_web_thread_sags_with_gravity_when_slack():
    th = make_thread((1, 1), (2, 1), (100, 300), (500, 300), 0.0, rest=600)
    for _ in range(120):
        simulate(th, np.array([100.0, 300.0]), np.array([500.0, 300.0]), DT, 900)
    assert th.pos[N // 2, 1] > 380
    np.testing.assert_allclose(th.pos[0], (100, 300))
    np.testing.assert_allclose(th.pos[-1], (500, 300))


def test_web_hand_leaving_drops_thread_and_clear_cuts():
    m = WebMode()
    m.update(_two_hands(1, (600, 300), 2, (605, 300)), DT, 0, SIZE)
    m.update(_two_hands(1, (500, 300), 2, (705, 300))[:1], DT, DT, SIZE)
    assert all(th.dying_since is not None for th in m.threads)
    m2 = WebMode()
    m2.update(_two_hands(1, (600, 300), 2, (605, 300)), DT, 0, SIZE)
    m2.clear()
    assert all(th.dying_since is not None for th in m2.threads)


# ---------------------------------------------------------------- count (finger calculator)
from airframe.modes import CountMode  # noqa: E402
from airframe.modes.count import COMMIT_SECONDS, FingerCounter, evaluate, fmt  # noqa: E402


def _hand_with(lm, pid, x):
    p = _hand_ptr(pid, (x, 400), False)
    shifted = lm.copy()
    shifted[:, 0] += x - 400
    p.hand.landmarks = shifted
    p.palm = shifted[[0, 5, 9, 13, 17], :2].mean(axis=0)
    return p


def _fingers(n: int) -> np.ndarray:
    """Synthetic hand showing ``n`` fingers (thumb counts last, like 1→index, 5→all)."""
    lm = make_landmarks(fist=True)
    open_lm = make_landmarks()
    order = [8, 12, 16, 20]
    for tip in order[: min(n, 4)]:
        lm[tip - 3: tip + 1] = open_lm[tip - 3: tip + 1]
    if n == 5:
        lm[1:5] = open_lm[1:5]
    return lm


def test_finger_counting_0_to_5():
    for n in range(6):
        assert metrics.count_fingers(_fingers(n)) == n, n


def test_ok_sign_is_not_a_fist():
    ok = make_landmarks(pinch=True)
    assert metrics.is_ok_sign(ok)
    assert not metrics.is_ok_sign(make_landmarks(fist=True))
    assert not metrics.is_ok_sign(_fingers(2))


def test_finger_counter_ignores_flicker():
    c = FingerCounter(window=9, switch=6)
    for _ in range(9):
        c.push(3)
    for n in (4, 3, 2, 3, 4):  # noisy frames
        assert c.push(n) == 3
    for _ in range(6):
        v = c.push(4)
    assert v == 4


def test_evaluate_and_format():
    assert evaluate(3, "+", 4) == 7 and evaluate(2, "−", 5) == -3
    assert evaluate(3, "×", 5) == 15 and evaluate(5, "÷", 2) == 2.5
    assert evaluate(4, "÷", 0) is None
    assert fmt(7.0) == "7" and fmt(2.5) == "2.5" and fmt(1 / 3) == "0.33" and fmt(None) == "∞"


def test_count_mode_live_equation_ok_sign_and_tape():
    m = CountMode()
    t = 0.0
    for _ in range(12):  # left shows 3, right shows 4
        t += DT
        m.update([_hand_with(_fingers(4), 2, 900), _hand_with(_fingers(3), 1, 300)], DT, t, SIZE)
    assert m.operands == [3, 4]
    assert m.equation() == ("3 + 4", 7)
    # 👌 twice with the right hand → + → − → ×   (the right operand stays frozen at 4 while signing)
    for sign in range(2):
        for _ in range(4):
            t += DT
            m.update([_hand_with(_fingers(3), 1, 300), _hand_with(make_landmarks(pinch=True), 2, 900)], DT, t, SIZE)
        assert m.op == ["−", "×"][sign] and m.operands == [3, 4]
        for _ in range(3):  # relax the hand between signs
            t += DT
            m.update([_hand_with(_fingers(3), 1, 300), _hand_with(_fingers(4), 2, 900)], DT, t, SIZE)
    # hold still → recorded once on the tape
    for _ in range(int(COMMIT_SECONDS * 30) + 15):
        t += DT
        m.update([_hand_with(_fingers(3), 1, 300), _hand_with(_fingers(4), 2, 900)], DT, t, SIZE)
    assert [e.text for e in m.tape] == ["3 × 4 = 12"]
    assert m.total == 12
    assert m.undo() == "UNDO" and m.tape == []


def test_count_mode_disables_v_sign_switch():
    assert CountMode.allows_mode_gesture is False
