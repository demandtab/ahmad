"""Quess Video #2 "The Perfect Move", professional reel cut (9:16, 35.0 s), built from scenes_pm (v2).

Reference treatment (client's reference trailer, motion only): headline top-centre at y=330, a different
hero in the centre of every beat, phone tilt-in then straighten, board push-ins, no orbit, logo scale-in, a 5 s
static end card, energetic Mixkit "Epic Games" (#76) with its drop on the 24.0 s climax.

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

# reel type: one Cinzel Black size (86 px at 1080) for every headline; gold underline w = 0.30 of the frame
T.NARR_SIZE = 172
T.RULE_W, T.RULE_H, T.RULE_GAP = 648, 6.0, 40
HEAD_CY = 660            # headline centre, y = 330 at 1080x1920
TOP_SAFE = 440           # 220 px

SRC2 = os.path.join(C.ROOT, 'src2')
CLEAN = os.environ.get('CLEAN') == '1'          # textless version

# ------------------------------------------------------------------ beat grid (SNAP): Mixkit "Epic Games" (#76)
BEATS = json.load(open(os.path.join(HERE, 'beats_76.json')))
MUSIC_FILE_OFFSET = BEATS['offset']               # track time at video 0.0 (its drop lands on 24.0)
SHIFT = 0.0
VBEATS = np.array(BEATS['grid'])                  # steady 105.5 BPM grid pinned on the drop


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
    return Seq(name, 60 if name in ('g1', 'tra', 'g1b') else 30)


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
N = {  # nominal cut times (snapped to the beat grid)
    'h1': 0.5, 'h2': 1.8, 'p1': 3.0, 'p1m': 5.0, 'p1a': 7.0, 'p2': 8.5, 'p2a': 10.5, 'p3': 12.5, 'hit': 14.5,
    'wct': 15.5, 'react': 17.5, 'l8': 19.0, 'c097': 20.5, 'fcp': 22.0, 'burn': 23.0, 'cx': 24.0, 'thr1': 25.5,
    'thr2': 26.75, 'logo': 28.0,
}
K = {k: S(v) for k, v in N.items()}
K['card'] = 30.0                                         # logo 27.97-30.0, end card 30.0-34.0 (no gap)
DURATION = 34.0
STOP = K['hit'] - 0.25
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


@functools.lru_cache(maxsize=1)
def stars():
    """The real starfield from the Quess outro, the reel's background throughout."""
    img = cv2.imread(os.path.join(SRC2, 'stills', 'bg_stars.png')).astype(np.float32) / 255.0
    return C.cover(img, zoom=1.25)


@functools.lru_cache(maxsize=None)
def still(name):
    return cv2.imread(os.path.join(SRC2, 'stills', name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


def arena_bg(t, ox=0.0, oy=0.0, dim=1.0):
    """Dark starry space with slow parallax (gold sparkles are added in render)."""
    M = np.float32([[1.04, 0, -W * 0.02 + P(ox) * 0.4], [0, 1.04, -H * 0.02 + P(oy) * 0.4 - P(20) * t]])
    return cv2.warpAffine(stars(), M, (W, H), borderMode=cv2.BORDER_REFLECT) * dim + 0.01


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
HERO_Y = 2000    # hero centre, y = 1000 at 1080x1920


def tilt_in(lt, d, yaw0, z0, push=1.10, settle=1.2):
    """UI motion graphic: the phone tilts in (yaw0 -> 0 deg over 1.2 s), then the camera pushes 1.00 -> push."""
    k = eo3(lt / settle)
    yaw = yaw0 * (1 - k)
    zoom = lerp(1.0, push, eio(seg(lt, settle, max(d, settle + 0.1))))
    return yaw, z0 / zoom


def shots():
    """(start, end, draw(local_t, t) -> frame, dissolve-in s). Hard cuts on the beat; no shot holds one angle > 3 s."""
    k_ = K
    out = []
    RED, GREEN, GREY = (1.0, 0.35, 0.2), (0.3, 1.0, 0.4), (0.8, 0.85, 0.95)

    # 1 HOOK: low angle on the real board, push-in 1.00 -> 1.08 per framing (fade in 0.25 handled in render)
    out.append((0.0, k_['h1'], lambda lt, t: frame_from('hook', 0.6 + lt * 0.85, (0.50, 0.55), lerp(1.0, 1.02, lt / 0.6)), 0))
    out.append((k_['h1'], k_['h2'], lambda lt, t: frame_from('hook', 1.0 + lt * 0.85, (0.47, 0.42),
                                                           lerp(1.12, 1.20, eio(lt / 1.1))), 0))
    out.append((k_['h2'], k_['p1'], lambda lt, t: frame_from('hook', 2.2 + lt * 0.85, (0.45, 0.70),
                                                           lerp(1.40, 1.51, eio(lt / 1.5))), 0))

    # 2 FIRE RED, BEFORE: phone tilts in (18 -> 0) and settles front-on, then the camera pushes in
    d = k_['p1m'] - k_['p1']
    def p1_before(lt, t):
        yaw, z = tilt_in(lt, d, 18, 1.95)
        return mock('g1', 0.25 + lt * (2.6 / d), [0, 0, z, yaw, 6, 0], (0.40, 0.52), (1080, HERO_Y), t, RED,
                    spot=(0.36, 0.5), spot_r=700, spot_k=0.25)
    out.append((k_['p1'], k_['p1m'], p1_before, 0))

    # 2 FIRE RED, MOVE: whole phone kept in frame (player names fully visible), slow push, spotlight on the piece
    d_m = k_['p1a'] - k_['p1m']
    def p1_move(lt, t):
        k = eio(lt / d_m)
        src = 2.85 + (lt * 0.9 if lt < 0.8 else 0.72 + (lt - 0.8) * (2.0 / max(d_m - 0.8, 0.1)))
        u = eio(seg(src, 3.9, 5.1))
        piece = (lerp(0.35, 0.45, u), lerp(0.47, 0.70, u))
        return mock('g1', src, [0, 0, lerp(1.76, 1.66, k), 0, 6, 0], (0.5, 0.5), (1080, HERO_Y), t, RED,
                    spot=piece, spot_r=600, spot_k=0.5)
    out.append((k_['p1m'], k_['p1a'], p1_move, 0))

    # --- zoomed animation clips, no device frame: red pieces, blue knight, overhead board (push/zoom)
    def anim(name, s0, sp, foc, z0, z1):
        def fn(lt, t, name=name, s0=s0, sp=sp, foc=foc, z0=z0, z1=z1):
            d = 1.8
            return frame_from(name, s0 + lt * sp, foc, lerp(z0, z1, eio(min(lt / d, 1.0))))
        return fn
    out.append((k_['p1a'], k_['p2'], anim('fcp', 0.1, 0.9, (0.5, 0.55), 1.0, 1.14), 0))       # MOVE PLAYED: FIRE RED pieces
    def knight(lt, t):
        f = cv2.resize(seq('c096').at(0.1 + lt * 0.8), (W, H), interpolation=cv2.INTER_AREA)
        return sharpen(C.cover(f, zoom=lerp(1.0, 1.14, eio(lt / 1.7)))) * 0.95 + 0.025
    out.append((k_['p2'], k_['p2a'], knight, 0))                                              # THEN PLAYER 2: blue knight
    out.append((k_['p2a'], k_['p3'], anim('top', 1.0, 0.6, (0.5, 0.56), 1.0, 1.10), 0))       # Ok... still fine: overhead

    # --- real gameplay, full board, no mockup (band over a blurred fill of the same footage, no dead space)
    def board(name, s0, sp, foc, z0, z1, d, pin_left=False):
        def fn(lt, t, name=name, s0=s0, sp=sp, foc=foc, z0=z0, z1=z1, d=d):
            z = lerp(z0, z1, eio(lt / d))
            fx = 0.5 / z if pin_left else foc[0]          # pin the left edge: the player list is never cut
            return frame_from(name, s0 + lt * sp, (fx, foc[1]), z, band=True, dy=P(460))
        return fn
    d6 = k_['hit'] - k_['p3']
    out.append((k_['p3'], k_['hit'], board('tra', 3.0, 2.3 / d6, (0.47, 0.50), 1.30, 1.40, d6), 0))        # AIR GREY approaches
    out.append((k_['hit'], k_['wct'], board('tra', 5.30, 0.29, (0.40, 0.46), 1.55, 1.60, 1.14), 0))         # WAIT.: the capture
    d7 = k_['react'] - k_['wct']
    out.append((k_['wct'], k_['react'], board('tra', 8.48, 0.2, (0.47, 0.50), 1.40, 1.30, d7), 0))          # cleared board
    # 7 cut-back to the FIRE RED player (no text)
    out.append((k_['react'], k_['l8'], lambda lt, t: frame_from('react', lt * 0.13, (0.40, 0.47),
                                                                lerp(1.0, 1.10, eio(lt / 1.5))), 0))

    # 8 YOUR PERFECT MOVE...: real gameplay, second half of Quess gameplay (1): AIR GREY pawn selected, moves +2
    d8 = k_['cx'] - k_['l8']
    out.append((k_['l8'], k_['cx'], board('g1b', 0.5, 3.1 / d8, (0.5, 0.52), 1.0, 1.10, d8, pin_left=True), 0))
    # climax: the rich low-angle board with the WATER BLUE portal (same recording), push
    out.append((k_['cx'], k_['thr1'], board('g1b', 6.4, 1.0, (0.5, 0.55), 1.05, 1.14, 1.7, pin_left=True), 0))
    # THREE OPPONENTS. / MORE THAN ONE MOVE TO WATCH.: the four players, two framings
    out.append((k_['thr1'], k_['thr2'], lambda lt, t: frame_from('thr', 1.8 + lt, (0.5, 0.62), lerp(1.0, 1.06, lt / 1.3)), 0))
    out.append((k_['thr2'], k_['logo'], lambda lt, t: frame_from('thr', 3.47 + lt * 0.76, (0.40, 0.58),
                                                                lerp(1.3, 1.38, lt / 1.3)), 0))

    # logo alone: the real Quess Q (from the outro), scale 0.6 -> 1.0 over 0.8 s, fade out 0.2 s
    def logo(lt, t):
        f = arena_bg(t) * 0.8
        lg = cv2.imread(os.path.join(SRC2, 'lock', 'quess_q_logo.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        d = K['card'] - K['logo']
        s_ = lerp(0.6, 1.0, eo3(lt / 0.8)) * 0.55 * W / lg.shape[1]      # full Q, w=0.55, never cropped
        a = clamp01(lt / 0.15)
        C.place(f, lg, W / 2, H * 900 / 1920, s_, a)
        return f
    out.append((k_['logo'], K['card'], logo, 0))

    # end card 30.0-35.0: real lockup (wordmark, AVAILABLE ON, badges, URL) fades in 0.3 s, then static
    def card(lt, t):
        f = arena_bg(0.0) * 0.8
        lk = still('lockup.png')
        a = 1.0                                  # straight cut from the logo: no black frame
        lk4 = np.dstack([lk[..., :3], np.ones(lk.shape[:2], np.float32)])
        C.place(f, lk4, W / 2, P(HERO_Y), W * 0.98 / lk.shape[1], a)
        return f
    out.append((K['card'], DURATION, card, 0))
    MOCK.update({3, 4})          # only two mockup shots (p1 before + move), 3.96 s in total
    return out


def ei(x):
    return C.ei3(x)


SHOTS = shots()

TEXT = [  # (t0, t1, kind, lines, fx): every text change is a hard cut on a beat
    (K['h1'], K['h2'], 'narr', ('WHEN YOU', 'FINALLY FIND'), 'reveal'),
    (K['h2'], K['p1'], 'narr', ('THE PERFECT', 'MOVE...'), 'reveal'),
    (K['p1'], K['p1a'], 'thought', ('I know exactly', 'what to do.'), 'reveal'),
    (K['p1a'], K['p2'], 'narr', ('MOVE PLAYED.',), 'glow'),
    (K['p2'], K['p2a'], 'narr', ('THEN PLAYER 2', 'MOVES.'), 'reveal'),
    (K['p2a'], K['p3'], 'thought', ('Ok... still fine.',), 'reveal'),
    (K['p3'], K['hit'], 'narr', ('THEN PLAYER 3', 'MOVES.'), 'reveal'),
    (K['hit'], K['wct'], 'narr', ('WAIT.',), 'hard'),
    (K['wct'], K['react'], 'narr', ('THAT', 'CHANGES', 'EVERYTHING.'), 'hard'),
    (K['l8'], K['cx'], 'narr', ('YOUR PERFECT', 'MOVE...'), 'wipe'),
    (K['cx'], K['thr1'], 'narr', ('JUST GOT', 'COMPLICATED.'), 'hard'),
    (K['thr1'], K['thr2'], 'narr', ('THREE', 'OPPONENTS.'), 'reveal'),
    (K['thr2'], K['logo'], 'narr', ('MORE THAN', 'ONE MOVE', 'TO WATCH.'), 'reveal'),
]
TURNS = [(K['p1'], 'FIRE RED'), (K['p2'], 'EARTH GREEN'), (K['p3'], 'AIR GREY')]
OPP_FLASH = [(K['thr1'], 'WATER BLUE'), (K['thr1'] + 0.4, 'AIR GREY'), (K['thr1'] + 0.8, 'EARTH GREEN')]
MONTAGE_HITS = [K['l8'], K['cx']]


@functools.lru_cache(maxsize=None)
def block(kind, lines):
    """Every headline: the same Cinzel Black size with the gold underline (thoughts in sentence case)."""
    img, am = T.narrator_img(lines)
    ink = (img[..., :3].mean(2) > 0.5) & (img[..., 3] > 0.5) & (am < 0.05)
    rows = np.nonzero(ink.any(1))[0]
    groups, s0, p = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - p > 3:
            groups.append((s0, p))
            s0 = r
        p = r
    groups.append((s0, p))
    return img, am, groups


def head_y(img, groups):
    """Image centre y that puts the letters' centre on y=330 (pushed down only if a block would enter the top zone)."""
    r0, r1 = groups[0][0], groups[-1][1]
    letters_c = (r0 + r1) / 2
    target = P(HEAD_CY)
    if target - (letters_c - r0) < P(TOP_SAFE + 30):
        target = P(TOP_SAFE + 30) + (letters_c - r0)
    return target - (letters_c - img.shape[0] / 2)


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
        cy = head_y(img, groups)
        if fx == 'hard':                       # sharp, clean, instant, no blur
            C.place(f, img, W / 2, cy)
            continue
        if fx == 'embers' and tau < 0.55:      # letters assemble from gold sparkles
            xs, ys, rnd = ink_points(kind, lines)
            k = eo3(tau / 0.55)
            px = W / 2 - w / 2 + xs + rnd[:, 0] * P(900) * (1 - k)
            py = cy - h / 2 + ys + rnd[:, 1] * P(500) * (1 - k) - P(200) * (1 - k)
            cols = np.tile(np.array([[0.40, 0.70, 1.0]], np.float32), (len(xs), 1)) * (0.25 + 0.75 * k) * 0.35
            C.splat(f, px, py, cols, sigma_core=0.8, sigma_glow=4.0, glow=0.5)
            if tau < 0.38:
                continue
            tau = (tau - 0.38) * 1.6
        im = img.copy()
        alpha = np.ones(h, np.float32)          # hard cut in: full opacity from the first frame, no crossfade
        scale = 1.0 + 0.08 * (1 - eo3(tau / 0.25))
        blur = 0.0
        if fx == 'wipe':                       # capture-slam wipe
            q = eo3(tau / 0.28)
            xs_ = np.arange(w, dtype=np.float32)
            im[..., 3] *= np.clip((q * (w + P(200)) - xs_) / P(60), 0, 1)[None, :]
            edge = np.exp(-((xs_ - q * (w + P(200)) + P(30)) / P(30)) ** 2) * (q < 1)
            im[..., :3] = np.clip(im[..., :3] + edge[None, :, None] * 0.8, 0, 1)
            scale, blur = 1.0, 0.0
        kr = eo3((tau - 0.12) / 0.32)          # gold underline draws outward, light catches it once
        xs_ = np.abs(np.arange(w) - w / 2)
        rule = am > 0
        im[..., 3] = np.where(rule, im[..., 3] * (xs_ <= kr * w / 2)[None, :], im[..., 3])
        qs = (tau - 0.30) / 0.4
        if 0 < qs < 1.1:
            shine = np.exp(-((np.arange(w) - qs * w) / (w * 0.06)) ** 2)
            im[..., :3] = np.where(rule[..., None], np.clip(im[..., :3] + shine[None, :, None] * 0.6, 0, 1.4), im[..., :3])
        im[..., 3] *= alpha[:, None]
        if blur > 0.03:
            kb = max(1, int(P(36) * blur))
            im = cv2.blur(im, (1, kb * 2 + 1))
        dx = P(8) * math.exp(-(tau - 0.05) / 0.05) * math.sin(tau * 90) if 0.05 < tau < 0.3 else 0.0
        if fx == 'glow':
            g = np.exp(-((np.arange(h) - h * 0.75) / (h * 0.5)) ** 2)[:, None] * clamp01(1 - tau / 0.6) * 0.5
            im[..., :3] = np.clip(im[..., :3] + g[..., None] * np.array([0.35, 0.65, 1.0], np.float32), 0, 1.5)
        C.place(f, im, W / 2 + dx, cy, scale)


@functools.lru_cache(maxsize=None)
def top_bed(k=0.42):
    """Darkening pooled behind the top headline zone (part of the vignette; no box)."""
    yy = np.arange(H, dtype=np.float32)
    prof = 1 - k * np.exp(-((yy - P(HEAD_CY)) / P(330)) ** 2)
    return prof[:, None, None].astype(np.float32)


@functools.lru_cache(maxsize=None)
def label_img():
    tm, _, _ = T._mask(('QUESS - 4 PLAYER CHESS',), 92)
    return T._compose(tm, np.zeros_like(tm), 0.7)


# ------------------------------------------------------------------ render
EMB = C.Embers(11, 90, size=(2, 5), speed=(60, 170), color=(0.40, 0.72, 1.0), down=3, blur=1.4, gain=0.6)
BOKEH = C.Embers(23, 14, size=(14, 30), speed=(40, 90), color=(0.35, 0.62, 0.95), down=4, blur=6.0, gain=0.22)


def shot_at(t):
    for i, (a, b, fn, tr) in enumerate(SHOTS):
        if a <= t < b or i == len(SHOTS) - 1:
            return i
    return len(SHOTS) - 1


def picture(t):
    i = shot_at(t)
    a, b, fn, _ = SHOTS[i]
    return fn(t - a, t)


def render(t):
    f = picture(t)
    if shot_at(t) in MOCK and shot_at(t - 0.5 / FPS) == shot_at(t):     # slight motion blur on the moving phone
        f = (f + picture(t - 0.5 / FPS)) * 0.5
    for t0, army in TURNS + OPP_FLASH:                                 # army-colour edge flash + sweep behind the headline
        if t0 <= t < t0 + 0.8:
            k = t - t0
            f += edge_mask() * (np.array(ARMY[army], np.float32) / 255.0) * 0.55 * math.exp(-k / 0.3)
            if (t0, army) in TURNS:
                T.army_sweep(f, ARMY[army], eio(k / 0.8), cy=HEAD_CY)
    for th in MONTAGE_HITS:
        if 0 <= t - th < 0.12:
            f = f * (1 + 0.5 * (1 - (t - th) / 0.12))
    hit = K['hit']
    if 0 <= t - hit < 0.4:                                             # capture burst: flash 0.08 + fade 0.17, shake
        k = t - hit
        f = f * (1 + 0.5 * (1.0 if k < 0.08 else max(0.0, 1 - (k - 0.08) / 0.17)))
        amp = P(14) * math.exp(-k / 0.12)
        M = np.float32([[1, 0, amp * math.sin(k * 70)], [0, 1, amp * 0.6 * math.cos(k * 53)]])
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    end = t >= K['card']
    if t < K['logo']:
        f *= top_bed()
    if not end:
        EMB.draw(f, t)
        BOKEH.draw(f, t, ox=-t * 30)
    f = C.grade(f, t, sat=1.12, leak=0.0 if end else 0.06, vig=0.3 if end else 0.62, grain=0.0 if end else 0.012,
                bloom=0.15 if end else 0.38)
    draw_text(f, t)
    if end and not CLEAN:                                             # end-card label, fades in with the lockup
        lk_h = still('lockup.png').shape[0] * W * 0.98 / still('lockup.png').shape[1]
        C.place(f, label_img(), W / 2, P(HERO_Y) - lk_h / 2 - P(40), 1.0, 1.0)
    if t < 0.25:                                                      # opening fade in 0.25 s
        f *= eio(t / 0.25)
    return f
