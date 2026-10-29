"""Quess Video #2 "The Perfect Move": picture timeline (9:16, ~34 s).

Reuses the trailer compositor (trailer/common.py: cover/band framing, embers, splat, bloom grade) and the
SNAP idea from trailer/scenes6.py: every cut and text hit is moved to the nearest music beat, rounded to a
frame. All pictures come from the real recordings cut by sources.py; the code only frames, times, grades and
adds light (flashes, sweeps, spotlights, embers) around the footage.

  SC=0.3333 python3 render_pm.py video out.mp4      # 720p
"""
import functools
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'trailer'))
import common as C  # noqa: E402
from common import P, W, H, FPS, eo3, eio, seg, lerp, clamp01  # noqa: E402
import type2 as T  # noqa: E402

SRC2 = os.path.join(C.ROOT, 'src2')
CLEAN = os.environ.get('CLEAN') == '1'          # textless version

# ------------------------------------------------------------------ beat grid (SNAP)
BEATS = json.load(open(os.path.join(HERE, 'beats_614.json')))
MUSIC_FILE_OFFSET = BEATS['offset']               # excerpt start inside Mixkit #614
LIFT_EXCERPT = 22.0                               # the track's lift (excerpt time), placed on line 8
LINE8 = 19.3                                      # nominal start of line 8 (payoff)
SHIFT = LIFT_EXCERPT - LINE8                      # music starts SHIFT s later in the excerpt
VBEATS = np.array([b - SHIFT for b in BEATS['beats'] if b - SHIFT > -0.01])


def S(t):
    """SNAP: nearest beat, rounded to the frame on which it appears."""
    b = VBEATS[np.argmin(np.abs(VBEATS - t))]
    return round(b * FPS) / FPS


# ------------------------------------------------------------------ sources
class Seq:
    def __init__(self, name, fps):
        self.dir = os.path.join(SRC2, name)
        self.fps = fps
        self.n = len([f for f in os.listdir(self.dir) if f.endswith('.jpg')])
        im = cv2.imread(os.path.join(self.dir, '0001.jpg'))
        h, w = im.shape[:2]
        self.k = min(1.0, max(1.6 * H / h, 1.7 * W / w)) if w > h else min(1.0, 1.4 * W / w)

    @functools.lru_cache(maxsize=6)
    def frame(self, i):
        i = int(min(max(i, 0), self.n - 1))
        img = cv2.imread(os.path.join(self.dir, f'{i + 1:04d}.jpg'))
        if self.k < 0.999:
            img = cv2.resize(img, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def at(self, t):
        return self.frame(int(round(t * self.fps)))


@functools.lru_cache(maxsize=None)
def seq(name):
    return Seq(name, 60 if name in ('g1', 'tra') else 30)


def sharpen(f, k=0.55):
    return f + (f - cv2.GaussianBlur(f, (0, 0), max(0.6, P(2.2)))) * k


def frame_from(name, src_t, focus, zoom, angle=0.0, band=False, dx=0.0, dy=0.0):
    img = seq(name).at(src_t)
    if band:   # landscape board kept whole over a blurred fill of the same footage, lifted above the text zone
        f = C.band(img, focus=focus, zoom=zoom, angle=angle, dx=dx, dy=-P(260) + dy, bg_dim=0.40, feather=160)
    else:
        f = C.cover(img, focus=focus, zoom=zoom, angle=angle, dx=dx, dy=dy)
    f = sharpen(f)
    return np.clip(f, 0, None) * 0.95 + 0.025      # lift shadows: no near-black frames


def to_out(name, focus, zoom, pt, band=False, dy=0.0):
    """Map a normalised source point to output pixels for a given framing (for spotlights)."""
    img = seq(name).frame(0)
    h, w = img.shape[:2]
    M = C.affine_cover(w, h, focus, zoom, 0.0, (W, H), 0.0, (-P(260) if band else 0.0) + dy,
                       mode='width' if band else 'cover')
    x, y = pt[0] * w, pt[1] * h
    return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]


def spotlight(f, cx, cy, r, k):
    yy, xx = np.ogrid[0:H, 0:W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    f *= (1 - k * np.clip(d - 1, 0, 1.4) / 1.4)[..., None]
    f += (0.10 * k * np.exp(-d ** 2))[..., None]
    return f


ARMY = {'FIRE RED': (40, 50, 235), 'EARTH GREEN': (60, 210, 70), 'AIR GREY': (215, 215, 215),
        'WATER BLUE': (240, 150, 30)}


@functools.lru_cache(maxsize=None)
def edge_mask():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    e = np.minimum.reduce([xx, yy, W - xx, H - yy]) / P(240)
    return np.exp(-e ** 2)[..., None]


# ------------------------------------------------------------------ timeline (nominal -> snapped)
N = {  # nominal cut times
    'p1': 3.0, 'p1m': 4.7, 'p1a': 7.3, 'p2': 8.6, 'p2a': 10.9, 'p3': 12.6, 'hit': 15.25, 'react': 18.0,
    'l8': LINE8, 'c097': 20.6, 'fcp': 21.8, 'burn': 22.7, 'qc': 23.6, 'end': 25.6, 'outro': 29.0,
    'h1': 0.45, 'h2': 1.75, 'thr2': 27.3,
}
K = {k: S(v) for k, v in N.items()}
K['wct'] = S(K['hit'] + 0.86)                            # "That changes everything." (next beat after the hit)
K['pq'] = S(K['outro'] + 2.6)                            # PLAY QUESS. (after the outro lockup settles)
DURATION = round((K['outro'] + 4.8) * FPS) / FPS
STOP = K['hit'] - 0.25                                   # 0.25 s stop-down before the hit


# ------------------------------------------------------------------ phone mockup in a dark 3D arena (trailer Device)
C.SRC_W, C.SRC_H = 1920, 1080          # the phone screen is 16:9; recordings are fitted inside it untouched


@functools.lru_cache(maxsize=1)
def phone():
    """The trailer's landscape phone, built at 1.7x texture resolution so punch-ins stay sharp."""
    sc = C.SC
    C.SC = sc * 1.7
    try:
        d = C.Device()
    finally:
        C.SC = sc
    return d


def fit_screen(name, src_t):
    """Real game screen fitted (not cropped) into the 1920x1080 phone screen; UI untouched."""
    img = seq(name).at(src_t)
    h, w = img.shape[:2]
    s_ = min(1920 / w, 1080 / h)
    nw, nh = int(round(w * s_)), int(round(h * s_))
    out = np.zeros((1080, 1920, 3), np.float32)
    x0, y0 = (1920 - nw) // 2, (1080 - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return sharpen(out, 0.45), (x0, y0, s_ / 1.0 * w, s_ * h)


def scr_pt(box, n):
    x0, y0, w, h = box
    return (x0 + n[0] * w, y0 + n[1] * h)


def arena_bg(t, ox=0.0, oy=0.0, dim=0.95):
    """The real (empty) Quess game room, defocused behind the phone, slow parallax."""
    img = seq('room').at(2.0 + t * 0.35)
    f = C.cover(img, focus=(0.5, 0.45), zoom=1.18, dx=P(ox), dy=P(oy))
    small = cv2.resize(f, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.8, P(9) / 6))
    return cv2.resize(small, (W, H)) * dim + 0.03


def device_shot(bg, screen, pose, glow=0.62, refl=0.45, rim_light=None, floor_dark=0.35):
    """trailer common.device_shot with a lighter, glossier floor (reflection + screen-light pool fill the frame)."""
    d = phone()
    frame = bg.copy()
    tex = d.texture(screen, glint=pose[3] * 0.012, env=cv2.resize(bg, (W // 8, H // 8)))
    Hm = d.homography(pose)
    warped = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a, rgb = warped[..., 3:4], warped[..., :3]
    fy = d.floor_y(pose, gap=0.08)
    x, y, z = pose[:3]
    horizon = C.proj(np.array([[x, fy, z]]))[0, 1]
    ys = np.arange(H, dtype=np.float32)
    k = np.clip((ys - (horizon - P(60))) / P(260), 0, 1)
    frame *= (1 - floor_dark * k)[:, None, None]
    wr = cv2.warpPerspective(tex, d.homography(pose, mirror_y=fy), (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    fade = np.clip(1 - (ys - horizon) / P(1400), 0, 1) ** 1.3 * (ys > horizon - P(4))
    ra = wr[..., 3:4] * (fade * refl)[:, None, None]
    frame[:] = frame * (1 - ra) + cv2.GaussianBlur(wr[..., :3], (0, 0), max(0.5, P(4))) * ra
    C.glow_add(frame, rgb * a, glow * 0.7, sigmas=(70, 220))
    small = cv2.resize(rgb * a, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
    pool = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(0.6, P(140) / 16)), (W, H))
    hy = int(max(0, min(H - 1, horizon)))
    span = H - hy
    if span > 0:
        src = pool[max(0, hy - span):hy][::-1]
        frame[hy:hy + src.shape[0]] += src * 0.7
    hull = cv2.convexHull(np.vstack([C.proj(d.outline(pose)), C.proj(d.outline(pose, d.th))]).astype(np.int32))
    side = 0.04 + 0.12 * np.array(rim_light)
    cv2.fillConvexPoly(frame, hull, tuple(float(v) for v in side), cv2.LINE_AA)
    frame[:] = frame * (1 - a) + rgb * a
    edge = cv2.morphologyEx(warped[..., 3], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    C.glow_add(frame, edge[..., None] * np.array(rim_light, np.float32), 1.0, sigmas=(6, 30), down=4)
    return frame, Hm


def mock(name, src_t, pose, aim_n, aim_xy, t, rim, spot=None, spot_r=520, spot_k=0.5):
    """Phone showing the real recording, floating in the arena: screen glow lights the room, glossy floor
    reflection, rim light. aim_n (normalised source point) lands on aim_xy (4K units)."""
    d = phone()
    scr, box = fit_screen(name, src_t)
    pose = d.aim(list(pose), scr_pt(box, aim_n), (P(aim_xy[0]), P(aim_xy[1])))
    bg = arena_bg(t, ox=-pose[3] * 14, oy=pose[4] * 6)
    f, Hm = device_shot(bg, scr, pose, rim_light=rim)
    if spot is not None:
        cx, cy = d.map_screen(Hm, [scr_pt(box, spot)])[0]
        f = spotlight(f, cx, cy, P(spot_r), spot_k)
    return np.clip(f, 0, None) * 0.96 + 0.02


MOCK = set()     # shots rendered with subframe motion blur


def shots():
    """(start, end, draw(local_t, t) -> frame, dissolve-in seconds). Hard cuts everywhere except one dissolve."""
    p1, p1m, p1a, p2, p2a, p3, hit = K['p1'], K['p1m'], K['p1a'], K['p2'], K['p2a'], K['p3'], K['hit']
    h1, h2, wct, thr2, pq = K['h1'], K['h2'], K['wct'], K['thr2'], K['pq']
    out = []
    RED, GREEN, GREY = (1.0, 0.35, 0.2), (0.3, 1.0, 0.4), (0.8, 0.85, 0.95)   # rim lights (RGB)

    # 1 HOOK in three framings, a hard cut on each text change: wide -> push on the character -> low over the pieces
    out.append((0.0, h1, lambda lt, t: frame_from('hook', 0.6 + lt * 0.85, (0.50, 0.52), lerp(1.0, 1.06, lt / h1)), 0))
    out.append((h1, h2, lambda lt, t: frame_from('hook', 1.0 + lt * 0.85, (0.47, 0.40),
                                                 lerp(1.20, 1.32, eio(lt / (h2 - h1))), angle=lerp(0, -1.0, lt)), 0))
    out.append((h2, p1, lambda lt, t: frame_from('hook', 2.2 + lt * 0.85, (0.45, 0.72),
                                                 lerp(1.45, 1.6, eio(lt / (p1 - h2))), angle=lerp(1.5, 0.5, lt)), 0))

    # 2 FIRE RED, BEFORE: phone floating in the arena, slow push-in, the piece is selected (real legal-move dots)
    d_b = p1m - p1
    def p1_before(lt, t):
        k = eio(lt / d_b)
        return mock('g1', 0.25 + lt * (2.6 / d_b), [0, 0, lerp(1.95, 1.75, k), lerp(20, 13, k), 10, 2],
                    (0.40, 0.52), (1080, 1450), t, RED, spot=(0.36, 0.5), spot_r=700, spot_k=0.3)
    out.append((p1, p1m, p1_before, 0))

    # 2 FIRE RED, MOVE: camera punches into the phone, slow motion on the fire-trail move, spotlight follows it
    d_m = p1a - p1m
    def p1_move(lt, t):
        k = eio(lt / d_m)
        src = 2.85 + (lt * 0.9 if lt < 1.0 else 0.9 + (lt - 1.0) * (1.85 / max(d_m - 1.0, 0.1)))
        u = eio(seg(src, 3.9, 5.1))
        piece = (lerp(0.35, 0.45, u), lerp(0.47, 0.70, u))
        return mock('g1', src, [0, 0, lerp(1.25, 0.82, k), lerp(9, 4, k), lerp(7, 4, k), lerp(-2, 0, k)],
                    (lerp(0.37, 0.41, k), lerp(0.50, 0.60, k)), (1080, 1700), t, RED,
                    spot=piece, spot_r=600, spot_k=0.55)
    out.append((p1m, p1a, p1_move, 0))

    # 3 MOVE PLAYED: low angle on the phone, steady hold, then a punch-in on the landed piece
    d_a = p2 - p1a
    def p1_after(lt, t):
        k = eio(seg(lt, d_a * 0.5, d_a))
        return mock('g1', min(5.6 + lt * 0.45, 6.3), [0, 0, lerp(1.85, 1.25, k), -16, -9, -2],
                    (lerp(0.42, 0.40, k), lerp(0.55, 0.62, k)), (1080, 1450), t, RED,
                    spot=(0.40, 0.62), spot_r=650, spot_k=0.35 * k)
    out.append((p1a, p2, p1_after, 0))

    # 4 EARTH GREEN: wide drift on the phone, then a punch-in on the move in slow motion
    d2 = p2a - p2
    def p2_move(lt, t):
        src = 0.05 + lt * 0.18 if lt < 0.8 else 0.194 + (lt - 0.8) * (1.0 / max(d2 - 0.8, 0.1))
        k = eio(seg(lt, 0.55, d2))
        u = seg(src, 0.25, 1.2)
        piece = (lerp(0.40, 0.44, u), lerp(0.46, 0.36, u))
        return mock('tra', src, [0, 0, lerp(2.0, 1.1, k), lerp(-22, -8, lt / d2), lerp(12, 7, k), lerp(-3, 0, lt / d2)],
                    (lerp(0.47, 0.42, k), lerp(0.50, 0.42, k)), (1080, 1450), t, GREEN,
                    spot=piece, spot_r=560, spot_k=0.5)
    out.append((p2, p2a, p2_move, 0))

    # 5 "Ok... still fine.": the changed board, tighter, phone turned the other way, uneasy drift
    d5 = p3 - p2a
    def p2_after(lt, t):
        k = lt / d5
        return mock('tra', 1.25 + lt * 0.95, [0, 0, lerp(1.45, 1.35, k), lerp(12, 7, k), 5, lerp(1.5, -1.0, k)],
                    (0.46, 0.50), (1080, 1450), t, GREEN)
    out.append((p2a, p3, p2_after, 0))

    # 6 AIR GREY: low angle under the phone, speed ramp into the capture, a 0.25 s stop-down hold
    d6 = hit - p3
    def p3_in(lt, t):
        k = lt / d6
        src = 3.0 + min(lt, d6 - 0.25) * (2.3 / max(d6 - 0.25, 0.1))
        f = mock('tra', src, [0, 0, lerp(1.8, 1.15, ei(k)), lerp(24, 12, k), lerp(-13, -6, k), lerp(4, 1, k)],
                 (lerp(0.42, 0.36, k), lerp(0.48, 0.42, k)), (1080, 1450), t, GREY)
        if lt > d6 - 0.25:
            f *= 0.8
        return f
    out.append((p3, hit, p3_in, 0))

    # 7 IMPACT "Wait.": the capture as it lands, slow motion, the burst only starting (board visible), sharp
    def impact(lt, t):
        return mock('tra', 5.30 + lt * 0.29, [0, 0, lerp(0.92, 0.88, lt), 2, 3, 0], (0.36, 0.44), (1080, 1600), t, GREY)
    out.append((hit, wct, impact, 0))

    # 7 "That changes everything.": hard cut to the cleared board, readable, slow pull-back
    def cleared(lt, t):
        d = K['react'] - wct
        return mock('tra', 8.48 + lt * 0.18, [0, 0, lerp(1.05, 1.35, eio(lt / d)), -6, 6, 0], (0.42, 0.48),
                    (1080, 1520), t, GREY, spot=(0.34, 0.42), spot_r=750, spot_k=0.3)
    out.append((wct, K['react'], cleared, 0))

    # 7 cut-back to the FIRE RED player (no text)
    def react(lt, t):
        return frame_from('react', lt * 0.35, (0.40, 0.47), lerp(1.0, 1.14, eio(lt / (K['l8'] - K['react']))))
    out.append((K['react'], K['l8'], react, 0))

    # 8 chaos montage: hard cuts on the beat
    def native(lt, t):
        f = seq('c096').at(0.2 + lt)
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        f = C.cover(f, zoom=lerp(1.08, 1.0, lt / 1.4))
        return sharpen(f) * 0.95 + 0.025
    out.append((K['l8'], K['c097'], native, 0))
    out.append((K['c097'], K['fcp'], lambda lt, t: frame_from('c097', 0.1 + lt, (0.56, 0.52), lerp(1.0, 1.1, lt)), 0))
    out.append((K['fcp'], K['burn'], lambda lt, t: frame_from('fcp', 0.1 + lt, (0.5, 0.55), lerp(1.12, 1.0, lt)), 0))
    out.append((K['burn'], K['qc'], lambda lt, t: frame_from('burn', 0.4 + lt * 1.4, (0.5, 0.5), 1.15), 0))
    d_qc = K['end'] - K['qc']
    out.append((K['qc'], K['end'], lambda lt, t: frame_from('qc', 0.3 + lt, (0.5, 0.55),
                                                             lerp(1.75, 1.0, eo3(lt / d_qc))), 0))

    # END: thrones in two framings (a hard cut on each END line)
    out.append((K['end'], thr2, lambda lt, t: frame_from('thr', 1.8 + lt, (0.5, 0.62), lerp(1.12, 1.04, lt / 1.7)), 0))
    out.append((thr2, K['outro'], lambda lt, t: frame_from('thr', 3.47 + lt * 0.76, (0.40, 0.58), lerp(1.45, 1.3, lt / 1.7)), 0))

    # real outro (the one dissolve); hard reframe on PLAY QUESS lifts the whole layer clear of the text
    def outro(lt, t):
        f = seq('outro').at(0.0 + lt)
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        lift = P(460) if t >= pq else 0.0
        M = np.float32([[1, 0, 0], [0, 1, -lift]])
        return cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
    out.append((K['outro'], DURATION, outro, 0.30))
    MOCK.update(range(3, 11))
    return out


def ei(x):
    return C.ei3(x)


SHOTS = shots()

# ------------------------------------------------------------------ text events
TEXT = [  # (t0, t1, kind, lines, fx): every text change is a hard cut on a beat (t0/t1 are cut times)
    (K['h1'], K['h2'], 'narr', ('WHEN YOU', 'FINALLY FIND'), 'embers'),
    (K['h2'], K['p1'], 'narr', ('THE PERFECT', 'MOVE...'), 'reveal'),
    (K['p1'], K['p1m'], 'thought', ('I know exactly', 'what to do.'), 'reveal'),        # on the BEFORE panel
    (K['p1a'], K['p2'], 'narr', ('MOVE PLAYED.',), 'glow'),
    (K['p2'], K['p2a'], 'narr', ('THEN PLAYER 2', 'MOVES.'), 'reveal'),
    (K['p2a'], K['p3'], 'thought', ('Ok... still fine.',), 'reveal'),
    (K['p3'], K['hit'], 'narr', ('THEN PLAYER 3', 'MOVES.'), 'reveal'),
    (K['hit'], K['wct'], 'thought', ('Wait.',), 'hard'),                               # on the capture panel
    (K['wct'], K['react'], 'thought', ('That changes', 'everything.'), 'hard'),
    (K['l8'], K['fcp'], 'narr', ('YOUR PERFECT', 'MOVE...'), 'wipe'),
    (K['fcp'], K['end'], 'narr', ('JUST GOT', 'COMPLICATED.'), 'reveal'),
    (K['end'], K['thr2'], 'narr', ('THREE', 'OPPONENTS.'), 'reveal'),
    (K['thr2'], K['outro'], 'narr', ('MORE THAN', 'ONE MOVE', 'TO WATCH.'), 'reveal'),
    (K['pq'], DURATION + 1, 'narr', ('PLAY QUESS.',), 'embers'),
]
TURNS = [(K['p1'], 'FIRE RED'), (K['p2'], 'EARTH GREEN'), (K['p3'], 'AIR GREY')]
MONTAGE_HITS = [K['l8'], K['c097'], K['fcp'], K['burn'], K['qc']]


@functools.lru_cache(maxsize=None)
def block(kind, lines):
    img, am = (T.narrator_img(lines) if kind == 'narr' else T.thought_img(lines))
    a = img[..., 3]
    ink = (img[..., :3].mean(2) > 0.5) & (a > 0.5)
    if am is not None:
        ink &= am < 0.05
    rows = np.nonzero(ink.any(1))[0]
    groups, s0, p = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - p > 3:
            groups.append((s0, p))
            s0 = r
        p = r
    groups.append((s0, p))
    return img, am, groups


@functools.lru_cache(maxsize=None)
def ink_points(kind, lines, n=900):
    img, _, _ = block(kind, lines)
    ys, xs = np.nonzero(img[..., 3] * (img[..., :3].mean(2) > 0.5) > 0.6)
    r = np.random.default_rng(len(lines[0]))
    i = r.choice(len(xs), size=min(n, len(xs)), replace=False)
    return xs[i].astype(np.float32), ys[i].astype(np.float32), r.uniform(-1, 1, (len(i), 2)).astype(np.float32)


def draw_text(f, t):
    if CLEAN:
        return
    for t0, t1, kind, lines, fx in TEXT:
        if not (t0 <= t < t1):
            continue
        tau = t - t0
        img, am, groups = block(kind, lines)
        h, w = img.shape[:2]
        out_k = 0.0                                     # hard out on the cut: the next block lands on the same frame
        if fx == 'hard':                       # line 7: sharp, clean, instant, no blur
            T.place_text(f, img)                    # hard in, hard out
            continue
        if fx == 'embers' and tau < 0.55:      # letters assemble from ember particles
            xs, ys, rnd = ink_points(kind, lines)
            k = eo3(tau / 0.55)
            sx = rnd[:, 0] * P(900) * (1 - k)
            sy = rnd[:, 1] * P(500) * (1 - k) + P(300) * (1 - k)
            hh = img.shape[0]
            top = T.anchor_row(img)
            cy = P(T.TEXT_BOTTOM) - (top - hh / 2)
            px = W / 2 - w / 2 + xs + sx
            py = cy - hh / 2 + ys + sy
            cols = np.tile(np.array([[0.55, 0.78, 1.0]], np.float32), (len(xs), 1)) * (0.25 + 0.75 * k) * 0.35
            C.splat(f, px, py, cols, sigma_core=0.8, sigma_glow=4.0, glow=0.5)
            if tau < 0.38:
                continue
            tau = (tau - 0.38) * 1.6
        # per-line reveal: each line scales in (1.12 -> 1.0) and goes blur -> sharp, staggered by 0.1 s
        im = img.copy()
        alpha = np.zeros(h, np.float32)
        scale = 1.0
        for i, (r0, r1) in enumerate(groups):
            ki = eo3((tau - i * 0.10) / 0.22)
            alpha[max(0, r0 - int(P(40))):r1 + int(P(40))] = ki
        alpha[groups[-1][1] + int(P(40)):] = eo3((tau - (len(groups) - 1) * 0.10) / 0.22)
        k0 = eo3(tau / 0.22)
        scale = 1.0 + 0.12 * (1 - k0)
        blur = 1 - eo3(tau / 0.26)
        if fx == 'wipe':                       # capture-slam wipe: hard edge sweeps across with a flash
            q = eo3(tau / 0.28)
            xs_ = np.arange(w, dtype=np.float32)
            colmask = np.clip((q * (w + P(200)) - xs_) / P(60), 0, 1)
            im[..., 3] *= colmask[None, :]
            edge = np.exp(-((xs_ - q * (w + P(200)) + P(30)) / P(30)) ** 2) * (q < 1)
            im[..., :3] = np.clip(im[..., :3] + edge[None, :, None] * 0.8, 0, 1)
            scale, blur = 1.0, 0.0
        if am is not None:                     # gold rule draws outward after the type lands, light catches it once
            kr = eo3((tau - 0.12) / 0.32)
            xs_ = np.abs(np.arange(w) - w / 2)
            keep = (xs_ <= kr * w / 2).astype(np.float32)
            rule = am > 0
            im[..., 3] = np.where(rule, im[..., 3] * keep[None, :], im[..., 3])
            qs = (tau - 0.30) / 0.4
            if 0 < qs < 1.1:
                shine = np.exp(-((np.arange(w) - qs * w) / (w * 0.06)) ** 2)
                im[..., :3] = np.where(rule[..., None], np.clip(im[..., :3] + shine[None, :, None] * 0.6, 0, 1.4), im[..., :3])
        im[..., 3] *= alpha[:, None] * (1 - out_k)
        if blur > 0.03:
            kb = max(1, int(P(36) * blur))
            im = cv2.blur(im, (1, kb * 2 + 1))
        dx = 0.0
        if fx in ('reveal', 'glow', 'embers') and 0.05 < tau < 0.3:       # small impact shake on the hit
            dx = P(8) * math.exp(-(tau - 0.05) / 0.05) * math.sin(tau * 90)
        if fx == 'glow':                        # emerges out of board glow
            g = np.exp(-((np.arange(h) - h * 0.75) / (h * 0.5)) ** 2)[:, None] * clamp01(1 - tau / 0.6) * 0.5
            im[..., :3] = np.clip(im[..., :3] + g[..., None] * np.array([0.35, 0.65, 1.0], np.float32), 0, 1.5)
        T.place_text(f, im, s=scale, dx=dx, ref=img)
        # thought text: soft dark backing already in the block; no glass/neon anywhere


# ------------------------------------------------------------------ render
EMB = C.Embers(11, 80, size=(2, 5), speed=(60, 170), color=(0.35, 0.6, 1.0), down=3, blur=1.6, gain=0.5)
BOKEH = C.Embers(23, 14, size=(14, 30), speed=(40, 90), color=(0.30, 0.55, 0.95), down=4, blur=6.0, gain=0.22)


def shot_at(t):
    for i, (a, b, fn, tr) in enumerate(SHOTS):
        if a <= t < b or i == len(SHOTS) - 1:
            return i
    return len(SHOTS) - 1


def picture(t):
    i = shot_at(t)
    a, b, fn, _ = SHOTS[i]
    f = fn(t - a, t)
    # beat-centred dissolve into the next shot (transition length stored on the next shot)
    if i + 1 < len(SHOTS):
        na, nb, nfn, ntr = SHOTS[i + 1]
        if ntr > 0 and t >= na - ntr / 2:
            k = eio((t - (na - ntr / 2)) / ntr)
            f = f * (1 - k) + nfn(t - na, t) * k
    if i > 0 and SHOTS[i][3] > 0 and t < a + SHOTS[i][3] / 2:
        pa, pb, pfn, _ = SHOTS[i - 1]
        k = eio((t - (a - SHOTS[i][3] / 2)) / SHOTS[i][3])
        f = pfn(t - pa, t) * (1 - k) + f * k
    return f


def render(t):
    f = picture(t)
    if shot_at(t) in MOCK and shot_at(t - 0.5 / FPS) == shot_at(t):     # slight motion blur on the moving phone
        f = (f + picture(t - 0.5 / FPS)) * 0.5
    # army-colour edge flash + light sweep behind the text zone on each player's turn
    for t0, army in TURNS:
        if t0 <= t < t0 + 1.0:
            k = t - t0
            f += edge_mask() * (np.array(ARMY[army], np.float32) / 255.0) * 0.55 * math.exp(-k / 0.3)
            T.army_sweep(f, ARMY[army], eio(k / 0.9))
    # montage hits: a short bright pop on each cut (built from the frame itself)
    for th in MONTAGE_HITS:
        if 0 <= t - th < 0.12:
            f = f * (1 + 0.6 * (1 - (t - th) / 0.12))
    # line 7: one impact flash from the real burst, then a short subtle shake (picture only here)
    hit = K['hit']
    if 0 <= t - hit < 0.45:
        k = t - hit
        f = f * (1 + 0.45 * math.exp(-k / 0.06))
        amp = P(14) * math.exp(-k / 0.12)
        M = np.float32([[1, 0, amp * math.sin(k * 70)], [0, 1, amp * 0.6 * math.cos(k * 53)]])
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    f *= T.text_bed(0.28)
    if t < K['outro'] + 0.3:
        EMB.draw(f, t)
        BOKEH.draw(f, t, ox=-t * 30)
    end_mix = seg(t, K['outro'] - 0.3, K['outro'] + 0.3)
    f = C.grade(f, t, sat=1.12, leak=0.06 * (1 - end_mix), vig=0.62 * (1 - end_mix * 0.6), grain=0.012,
                bloom=0.38 * (1 - end_mix))
    draw_text(f, t)
    if not CLEAN and t >= K['outro'] + 2.1:
        a = eio((t - K['outro'] - 2.1) / 0.4)
        T.place_text(f, info_img(), bottom=2690, a=a)
    return f


@functools.lru_cache(maxsize=None)
def info_img():
    """End-card info under the outro's real badges and URL: same font (Cinzel Black), small info size."""
    tm, _, _ = T._mask(('QUESS - 4 PLAYER CHESS', 'An internet connection is needed to play.'), 70, gap=0.8)
    return T._compose(tm, np.zeros_like(tm), 0.7)
