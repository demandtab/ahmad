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
BEATS = json.load(open(os.path.join(HERE, 'beats_706.json')))   # Mixkit "Games Music" (#706), 139.7 BPM
MUSIC_FILE_OFFSET = BEATS['offset']               # track time at video 0.0 (its lift lands on 24.0)
SHIFT = 0.0
VBEATS = np.array(BEATS['grid'])                  # steady grid pinned on the lift


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
        """Frame-blended sampling: slowed clips stay smooth at a constant 30 fps (no repeated-frame judder)."""
        x = max(0.0, t * self.fps)
        i = int(x)
        k = x - i
        if k < 0.02 or i + 1 >= self.n:
            return self.frame(i)
        return self.frame(i) * (1 - k) + self.frame(i + 1) * k


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
    'h1': 0.4, 'h2': 1.7, 'p1': 3.0, 'p1a': 6.9, 'p2': 8.6, 'p2a': 10.3, 'join': 12.4, 'p3': 15.9, 'hit': 17.6,
    'wct': 18.85, 'l8': 20.6, 'cx': 24.0, 'thr1': 25.7, 'thr2': 27.0, 'share': 28.3, 'logo': 30.4,
}
K = {k: S(v) for k, v in N.items()}
K['zero'] = 0.0
K['card'] = K['logo'] + 2.0                              # logo 2 s, then the end card 4 s (no gap)
DURATION = round((K['card'] + 4.0) * FPS) / FPS
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


def remap_t(lt, keys):
    return C.remap(lt, keys)


def burst(f, cx, cy, k, col, r=260, gain=0.9):
    """Soft landing / capture glow: a radial bloom in the army colour that eases out (no flicker)."""
    if k <= 0 or k >= 1:
        return
    yy, xx = np.ogrid[0:H, 0:W]
    rr = P(r) * (0.6 + 0.8 * eo3(k))
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / rr ** 2
    a = math.sin(math.pi * min(1.0, k * 1.6)) ** 2 if k < 0.31 else (1 - k) / 0.69
    f += (np.exp(-d2 * 2.2) * a * gain)[..., None] * (np.array(col, np.float32) / 255.0)


def full(name, src, foc, z):
    """Gameplay full-screen (9:16 cover): no inset, no device frame, no empty bands."""
    return frame_from(name, src, foc, z)


BOARD = dict(x0=0.395, x1=0.645, y0=0.355, y1=0.725, n=14)   # real top-down board in the 'top' clip (source coords)


def cell(c, r):
    return (BOARD['x0'] + (c + 0.5) * (BOARD['x1'] - BOARD['x0']) / BOARD['n'],
            BOARD['y0'] + (r + 0.5) * (BOARD['y1'] - BOARD['y0']) / BOARD['n'])


PIECE = {'EARTH GREEN': 'piece_earth_green.png', 'FIRE RED': 'piece_fire_red.png',
         'WATER BLUE': 'piece_water_blue.png', 'AIR GREY': 'piece_air_grey.png'}
# (start key, end key, spec): spec = army, from cell, to cell, slide start, src t0, focus, zoom0, zoom1, [static pieces], [bursts]
HOOK = dict(army='EARTH GREEN', a=(6, 0), b=(6, 4), ts=0.45, src=0.2, foc=(0.52, 0.54), z=(1.36, 1.48),
            static=[('FIRE RED', (9, 5))], bursts=[('FIRE RED', (9, 5), 1.75)])
HOOK_HIT = 1.75                                            # red glow burst (music hit SFX)
MOVE_SLIDE = 0.8
BOARD_CY = 1150                                            # board centre, y in 1080x1920 (below the headline)


@functools.lru_cache(maxsize=None)
def board_mask_out(foc, z, dy):
    """Soft mask of the real board in output space (1 on the board): the surroundings are darkened, not shown."""
    m = np.zeros((H, W), np.float32)
    pts = [to_out('top', foc, z, (x, y), dy=dy) for x, y in ((BOARD['x0'], BOARD['y0']), (BOARD['x1'], BOARD['y0']),
                                                              (BOARD['x1'], BOARD['y1']), (BOARD['x0'], BOARD['y1']))]
    cv2.fillConvexPoly(m, np.int32([[p[0], p[1]] for p in pts]), 1.0)
    return cv2.GaussianBlur(m, (0, 0), P(40))[..., None]


def hook_shot(spec, d):
    """0-3 s: the real top-down board close and full-frame; one green piece slides (eased, glow trail), the red
    piece takes a glow burst; slow push-in. Board centred below the headline; surroundings darkened."""
    def fn(lt, t):
        U = W / 1080.0
        dy = P(2 * BOARD_CY) - H / 2
        k = (lt - spec['ts']) / MOVE_SLIDE
        e = eio(clamp01(k))
        z = lerp(spec['z'][0], spec['z'][1], eio(lt / d))
        foc = spec['foc']
        f = frame_from('top', spec['src'] + lt * 0.5, foc, z, dy=dy)
        mk = board_mask_out(foc, round(z, 2), dy)
        f = f * (0.30 + 0.70 * mk)
        hpx = int(200 * U * z / 1.36)
        for army, c in spec.get('static', []):
            x, y = to_out('top', foc, z, cell(*c), dy=dy)
            C.place(f, piece_sprite(PIECE[army], hpx), x, y - hpx * 0.38, 1.0, 1.0)
        for army, c, tb in spec.get('bursts', []):
            x, y = to_out('top', foc, z, cell(*c), dy=dy)
            burst(f, x, y - hpx * 0.2, (lt - tb) / 0.8, ARMY[army], r=320, gain=1.0)
        pa, pb = to_out('top', foc, z, cell(*spec['a']), dy=dy), to_out('top', foc, z, cell(*spec['b']), dy=dy)
        spr = piece_sprite(PIECE[spec['army']], hpx)
        col = ARMY[spec['army']]
        if 0 < k < 1:
            for j, back in enumerate((0.16, 0.10, 0.05)):
                kb = eio(clamp01(k - back / MOVE_SLIDE))
                g = spr.copy()
                g[..., :3] = g[..., :3] * 0.3 + np.array(col, np.float32) / 255 * 0.7
                C.place(f, g, lerp(pa[0], pb[0], kb), lerp(pa[1], pb[1], kb) - hpx * 0.38, 1.0, 0.26 - j * 0.07)
        x, y = lerp(pa[0], pb[0], e), lerp(pa[1], pb[1], e)
        burst(f, x, y - hpx * 0.1, (k - 1.0) * MOVE_SLIDE / 0.7, col, r=260, gain=0.75)
        C.place(f, spr, x, y - hpx * 0.38, 1.0, 1.0)
        return f
    return fn


def clip_move(name, s0, sp, moves, d, col, smooth=False):
    """A real animation clip, full-frame. Each move: the camera slides into a new framing (eased, with a short
    motion trail), lands with a soft glow burst on the piece (+ tick), then a slow push-in.
    moves = [(t_start, (fx, fy), zoom_from, zoom_to, dx_from (1080 px), piece_xy (1080 coords))]."""
    def fn(lt, t):
        U = W / 1080.0
        i = max(j for j, m in enumerate(moves) if lt >= m[0])
        t0, foc, z0, z1, dx0, pxy = moves[i]
        t1 = moves[i + 1][0] if i + 1 < len(moves) else d
        u = lt - t0
        ks = eio(clamp01(u / 0.6))                                      # the slide
        z = lerp(z0, z1, eio(clamp01(u / max(0.1, t1 - t0))))
        def at(tt, kk):
            return frame_from(name, s0 + tt * sp, foc, z, dx=P(2 * dx0) * (1 - kk))
        f = at(lt, ks)
        if smooth and u >= 0.6:
            f = f * 0.65 + at(lt - 1.0 / FPS, 1.0) * 0.35
        return f
    return fn


def gameplay(name, s0, keys, foc, z0, z1, d, glow=None, pin_left=False):
    """Real recorded gameplay with the whole board across the frame (armies not cut): board band centred under the
    headline, filled above/below by the same footage (blurred), push-in on the move, glow on the landing."""
    def fn(lt, t):
        src = C.remap(lt, keys) if keys else s0 + lt
        z = lerp(z0, z1, eio(lt / d))
        fx = 0.5 / z if pin_left else foc[0]
        img = seq(name).at(src)
        dy = P(2 * BOARD_CY) - H / 2 - P(260) * 0 
        f = C.band(img, focus=(fx, foc[1]), zoom=z, dx=0.0, dy=dy, bg_dim=0.62, feather=200)
        f = sharpen(f) * 0.95 + 0.025
        if glow:
            gx, gy, tl, col = glow
            ih, iw = img.shape[:2]
            M = C.affine_cover(iw, ih, (fx, foc[1]), z, 0.0, (W, H), 0.0, dy, mode='width')
            x = M[0, 0] * gx * iw + M[0, 1] * gy * ih + M[0, 2]
            y = M[1, 0] * gx * iw + M[1, 1] * gy * ih + M[1, 2]
            burst(f, x, y, (lt - tl) / 0.8, col, r=230, gain=0.65)
        return f
    return fn


def board_on_stars(t, src_t, z, foc=(0.52, 0.54), pieces=(), slide=None):
    """The real top-down board (Game Quess Env) full width and centred, set on the reel's starry background:
    the black space above and the red floor below are replaced by stars (no army cut, no bars).
    pieces: [(army, cell)] placed still; slide: (army, cell_a, cell_b, k) one eased slide, no trail, no glow."""
    U = W / 1080.0
    dy = P(2 * BOARD_CY) - H / 2
    f = frame_from('top', src_t, foc, z, dy=dy)
    mk = board_mask_out(foc, round(z, 2), dy)
    f = f * mk + arena_bg(t) * 0.85 * (1 - mk)
    hpx = int(190 * U * z / 1.27)
    for army, c in pieces:
        x, y = to_out('top', foc, z, cell(*c), dy=dy)
        C.place(f, piece_sprite(PIECE[army], hpx), x, y - hpx * 0.38, 1.0, 1.0)
    if slide:
        army, ca, cb, k = slide
        pa, pb = to_out('top', foc, z, cell(*ca), dy=dy), to_out('top', foc, z, cell(*cb), dy=dy)
        e = eio(clamp01(k))
        C.place(f, piece_sprite(PIECE[army], hpx), lerp(pa[0], pb[0], e), lerp(pa[1], pb[1], e) - hpx * 0.38, 1.0, 1.0)
    return f


def board_beat(army, ca, cb, ts, d, z0=1.20, z1=1.26):
    """One beat on the real board: one piece slides (ease-in-out, 0.8 s), lands on its square (tick), slow push-in."""
    def fn(lt, t):
        z = lerp(z0, z1, eio(lt / d))
        return board_on_stars(t, 0.6 + lt * 0.5, z, slide=(army, ca, cb, (lt - ts) / MOVE_SLIDE))
    return fn


def real(name, keys, foc, z0, z1, d, land=None):
    """A real recorded move, full frame (9:16 cover), centred on the moving piece. Zoom >= 1.2 with the focus low
    enough that the HUD (player list, timers, camera/settings icons) is outside the frame. No added effects."""
    def fn(lt, t):
        z = lerp(z0, z1, eio(lt / d))
        return full(name, C.remap(lt, keys), foc, z)
    return fn


SLIDES = []   # (start key, slide start in the beat) -> landing ticks


def shots():
    """(start, end, draw(local_t, t) -> frame, dissolve-in s). Hard cuts on the beat."""
    k_ = K
    out = []

    # 1 HOOK + 2-5: the real top-down board, full-screen; each clip one piece slides one move (eased, glow trail),
    #   lands with a glow burst (+ tick); the camera pushes in on the move. Different framing per clip.
    out.append((0.0, k_['p1'], lambda lt, t: board_on_stars(t, 0.2 + lt * 0.4, lerp(1.20, 1.26, eio(lt / k_['p1']))), 0))
    G, R, B = ARMY['EARTH GREEN'], ARMY['FIRE RED'], ARMY['WATER BLUE']
    d = k_['p1a'] - k_['p1']                    # green: EARTH GREEN piece slides (Quess Gameplay tra)
    out.append((k_['p1'], k_['p1a'], real('tra', [(0.0, 0.05), (d, 1.9)], (0.43, 0.55), 1.22, 1.34, d), 0))
    d = k_['p2'] - k_['p1a']                    # red: FIRE RED fire-trail move (Quess gameplay (1))
    out.append((k_['p1a'], k_['p2'], real('g1', [(0.0, 3.3), (d, 5.45)], (0.41, 0.62), 1.22, 1.32, d), 0))
    d = k_['p2a'] - k_['p2']                    # AIR GREY pawn slides two squares (Quess gameplay (1), later)
    out.append((k_['p2'], k_['p2a'], real('g1b', [(0.0, 1.0), (d, 3.0)], (0.49, 0.62), 1.22, 1.30, d), 0))
    # blue glass knight on its board (kept): eased camera move + tick, no trail/burst
    out.append((k_['p2a'], k_['join'], clip_move('c096f', 0.3, 0.6, [(0.0, (0.5, 0.55), 1.14, 1.04, -300, (520, 1000))],
                                                 k_['join'] - k_['p2a'], B), 0))

    # THEY JOIN: pieces assemble on the board
    out.append((k_['join'], k_['p3'], assemble, 0))

    # 15.8-25.7 s: real recordings, full frame, a different moment per beat, no added pieces
    dd = k_['hit'] - k_['p3']
    out.append((k_['p3'], k_['hit'], real('tra', [(0.0, 3.0), (dd, 5.28)], (0.45, 0.55), 1.22, 1.34, dd), 0))
    dd = k_['wct'] - k_['hit']
    out.append((k_['hit'], k_['wct'], real('tra', [(0.0, 5.30), (dd, 5.88)], (0.38, 0.52), 1.34, 1.40, dd), 0))
    dd = k_['l8'] - k_['wct']
    out.append((k_['wct'], k_['l8'], real('g1b', [(0.0, 3.0), (dd, 4.6)], (0.50, 0.62), 1.20, 1.28, dd), 0))
    dd = k_['cx'] - k_['l8']
    out.append((k_['l8'], k_['cx'], real('g1', [(0.0, 0.3), (dd, 2.85)], (0.40, 0.60), 1.22, 1.34, dd), 0))
    dd = k_['thr1'] - k_['cx']
    out.append((k_['cx'], k_['thr1'], real('g1b', [(0.0, 6.4), (dd, 7.6)], (0.55, 0.55), 1.20, 1.28, dd), 0))
    # THREE OPPONENTS.: the real overhead board, all four armies; MORE THAN ONE MOVE: four armies at start
    dd = k_['thr2'] - k_['thr1']
    out.append((k_['thr1'], k_['thr2'], real('qvo', [(0.0, 0.3), (dd, 1.6)], (0.40, 0.50), 1.0, 1.06, dd), 0))
    dd = k_['share'] - k_['thr2']
    out.append((k_['thr2'], k_['share'], real('qc', [(0.0, 0.3), (dd, 1.6)], (0.46, 0.55), 1.0, 1.06, dd), 0))
    # SHARE IT WITH YOUR FRIENDS: clean starry CTA, three equal circles pop in one by one, names below
    out.append((k_['share'], k_['logo'], share, 0))

    # logo alone: the full Quess Q, scale 0.6 -> 1.0 over 0.8 s
    def logo(lt, t):
        f = arena_bg(t) * 0.8
        lg = still_lock('quess_q_logo.png')
        s_ = lerp(0.6, 1.0, eo3(lt / 0.8)) * 0.55 * W / lg.shape[1]
        C.place(f, lg, W / 2, H * 900 / 1920, s_, clamp01(lt / 0.15))
        return f
    out.append((k_['logo'], K['card'], logo, 0))

    def card(lt, t):
        f = arena_bg(0.0) * 0.8
        lk = still('lockup.png')
        lk4 = np.dstack([lk[..., :3], np.ones(lk.shape[:2], np.float32)])
        C.place(f, lk4, W / 2, P(HERO_Y), W * 0.98 / lk.shape[1], 1.0)
        return f
    out.append((K['card'], DURATION, card, 0))
    return out


@functools.lru_cache(maxsize=None)
def still_lock(name):
    return cv2.imread(os.path.join(SRC2, 'lock', name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


@functools.lru_cache(maxsize=None)
def piece_sprite(name, hpx):
    im = still(name)
    s_ = hpx / im.shape[0]
    return cv2.resize(im, (max(2, int(im.shape[1] * s_)), int(hpx)), interpolation=cv2.INTER_AREA)


@functools.lru_cache(maxsize=None)
def tag_img(lines, size=34, backing=False):
    tm, _, _ = T._mask(tuple(lines), size * 2, gap=0.35)
    if not backing:
        return T._compose(tm, np.zeros_like(tm), 0.8)
    back = np.zeros_like(tm)
    ys, xs = np.nonzero(tm > 0.1)
    k = int(P(30))
    cv2.rectangle(back, (int(xs.min()) - 2 * k, int(ys.min()) - k), (int(xs.max()) + 2 * k, int(ys.max()) + k), 1, -1)
    back = cv2.GaussianBlur(back, (0, 0), P(40)) * 0.7
    img = T._compose(tm, np.zeros_like(tm), 0.9, extra_alpha=back)
    img[..., :3] = np.clip(img[..., :3] * 1.08, 0, 1)
    return img


JOIN = [  # (sprite, name, army, colour, start (s into the beat), from (1080 coords), to)
    ('piece_fire_red.png', 'TELLER', 'FIRE RED', ARMY['FIRE RED'], 0.30, (-260, 480), (300, 780)),
    ('piece_earth_green.png', 'JOHN GREEN', 'EARTH GREEN', ARMY['EARTH GREEN'], 0.85, (1340, 480), (720, 780)),
    ('piece_water_blue.png', 'JACKY', 'WATER BLUE', ARMY['WATER BLUE'], 1.40, (-260, 1750), (300, 1190)),
    ('piece_air_grey.png', 'OPIE', 'AIR GREY', ARMY['AIR GREY'], 1.95, (1340, 1750), (720, 1190)),
]
SLIDE = 0.5


CARDS = [('card_teller_fire_red.png', 'TELLER', 'FIRE RED', 318, 770, 0.25),
         ('card_john_green_earth_green.png', 'JOHN GREEN', 'EARTH GREEN', 762, 770, 0.85),
         ('card_jacky_water_blue.png', 'JACKY', 'WATER BLUE', 318, 1300, 1.45),
         ('card_opie_air_grey.png', 'OPIE', 'AIR GREY', 762, 1300, 2.05)]
CARD_W = 424          # px at 1080: all four cards equal


@functools.lru_cache(maxsize=None)
def card_img(name, px):
    im = still_lock(name)[..., :3]
    h, w = im.shape[:2]
    cxs = {'card_teller_fire_red.png': 0.51, 'card_john_green_earth_green.png': 0.45,
           'card_jacky_water_blue.png': 0.60, 'card_opie_air_grey.png': 0.70}
    cw = int(0.62 * w)                                                 # whole figure (full height), centred
    x0 = int(min(max(cxs[name] * w - cw / 2, 0), w - cw))
    col = cv2.resize(im[:, x0:x0 + cw], (int(px * cw / h), px), interpolation=cv2.INTER_AREA)
    im = np.zeros((px, px, 3), np.float32)
    ox = (px - col.shape[1]) // 2
    im[:, ox:ox + col.shape[1]] = col
    xs = np.abs(np.arange(px, dtype=np.float32) / px - 0.5)
    side = np.clip(1 - (xs - 0.20) / 0.11, 0, 1) ** 1.5               # neighbours at the edges fade to dark
    im = np.clip(im * 1.25 + 0.02, 0, 1) * (0.06 + 0.94 * side)[None, :, None]
    gold = np.array(C.GOLD_RGB[::-1], np.float32) / 255
    b = max(2, px // 90)
    im[:b], im[-b:], im[:, :b], im[:, -b:] = gold, gold, gold, gold          # thin gold frame
    return np.dstack([im, np.ones((px, px), np.float32)]).astype(np.float32)


WHITE = []          # labels composited after the grade (clean white, not toned)
SPARK_J = C.Embers(57, 40, size=(2, 4), speed=(30, 70), color=(0.45, 0.75, 1.0), down=3, blur=1.0, gain=0.5)


def assemble(lt, t):
    """THEY JOIN: four equal player cards (2x2) pop in one by one (TELLER, JOHN GREEN, JACKY, OPIE), each with its
    name and army below; a soft gold shine sweeps across each card once; faint gold sparkles in the background."""
    U = W / 1080.0
    f = arena_bg(t) * 0.85
    sp = np.zeros_like(f)
    SPARK_J.draw(sp, t)
    f += sp * 0.6
    px = int(CARD_W * U)
    for name, who, army, x, y, ts in CARDS:
        k = (lt - ts) / 0.3
        if k <= 0:
            continue
        sc = 1 + 1.7 * (min(k, 1) - 1) ** 3 + 0.7 * (min(k, 1) - 1) ** 2      # pop-in (ease-out-back)
        img = card_img(name, px).copy()
        q = (lt - ts - 0.35) / 0.6                                          # one gentle shine sweep
        if 0 < q < 1.2:
            xs = np.arange(px, dtype=np.float32)[None, :] + np.arange(px, dtype=np.float32)[:, None] * 0.5
            band = np.exp(-((xs - (q * 1.4 - 0.2) * px * 1.5) / (px * 0.08)) ** 2) * 0.35
            img[..., :3] = np.clip(img[..., :3] + band[..., None] * np.array([0.6, 0.85, 1.0], np.float32), 0, 1)
        C.place(f, img, x * U, y * U, max(0.05, sc), clamp01(k * 2))
        if not CLEAN:
            a = clamp01((lt - ts - 0.25) / 0.25)
            WHITE.append((tag_img((who,), 42, True), x * U, (y + 246) * U, a))
            WHITE.append((tag_img((army,), 34, True), x * U, (y + 294) * U, a))
    return f


SHARE = [('card_teller_fire_red.png', 'TELLER', 300, 780), ('card_john_green_earth_green.png', 'JOHN GREEN', 750, 780),
         ('card_jacky_water_blue.png', 'JACKY', 300, 1290), ('card_opie_air_grey.png', 'OPIE', 750, 1290)]


@functools.lru_cache(maxsize=None)
def avatar(name, dpx):
    im = still_lock(name)
    h, w = im.shape[:2]
    side = int(0.46 * w)                                                 # head and shoulders of this player
    cx = {'card_teller_fire_red.png': 0.51, 'card_john_green_earth_green.png': 0.47,
          'card_jacky_water_blue.png': 0.58, 'card_opie_air_grey.png': 0.68}[name]
    x0 = int(min(max(cx * w - side / 2, 0), w - side))
    y0 = int(0.06 * h)
    c = cv2.resize(im[y0:y0 + side, x0:x0 + side, :3], (dpx, dpx), interpolation=cv2.INTER_AREA)
    c = np.clip(c * 1.45 + 0.02, 0, 1)                                   # brighter, never washed out
    c = np.clip(c + (c - cv2.GaussianBlur(c, (0, 0), 1.5)) * 0.6, 0, 1)
    yy, xx = np.mgrid[0:dpx, 0:dpx].astype(np.float32)
    r = np.sqrt((xx - dpx / 2) ** 2 + (yy - dpx / 2) ** 2)
    disc = np.clip(dpx / 2 - 2 - r, 0, 1)
    ring = np.clip(1 - np.abs(r - (dpx / 2 - 3)) / 2.5, 0, 1)
    gold = np.array(C.GOLD_RGB[::-1], np.float32) / 255
    rgb = c * (1 - ring[..., None]) + gold * ring[..., None]
    return np.dstack([rgb, np.maximum(disc, ring)]).astype(np.float32)


def share(lt, t):
    """SHARE IT WITH YOUR FRIENDS: clean dark starry background, four equal circles pop in one by one
    (TELLER, JOHN GREEN, JACKY, OPIE), each with its name clearly below."""
    f = arena_bg(t) * 0.85
    U = W / 1080.0
    d = int(430 * U)
    for i, (name, who, x, y) in enumerate(SHARE):
        k = (lt - 0.05 - i * 0.1) / 0.25
        if k <= 0:
            continue
        sc = 1 + 1.7 * (min(k, 1) - 1) ** 3 + 0.7 * (min(k, 1) - 1) ** 2
        C.place(f, avatar(name, d), x * U, y * U, max(0.05, sc), clamp01(k * 2))
        if not CLEAN:
            WHITE.append((tag_img((who,), 50, True), x * U, (y + 268) * U, clamp01((k - 0.5) * 2.5)))
    return f


def ei(x):
    return C.ei3(x)


SHOTS = shots()

TEXT = [  # (t0, t1, kind, lines, fx): every text change is a hard cut on a beat
    (K['h1'], K['h2'], 'narr', ('WHEN YOU', 'FINALLY FIND'), 'reveal'),
    (K['h2'], K['p1'], 'narr', ('THE PERFECT', 'MOVE...'), 'reveal'),
    (K['p1'], K['p1a'], 'thought', ('I know exactly', 'what to do.'), 'reveal'),
    (K['p1a'], K['p2'], 'narr', ('MOVE PLAYED.',), 'reveal'),
    (K['p2'], K['p2a'], 'narr', ('THEN PLAYER 2', 'MOVES.'), 'reveal'),
    (K['p2a'], K['join'], 'thought', ('Ok... still fine.',), 'reveal'),
    (K['join'], K['p3'], 'narr', ('THEY JOIN',), 'reveal'),
    (K['p3'], K['hit'], 'narr', ('THEN PLAYER 3', 'MOVES.'), 'reveal'),
    (K['hit'], K['wct'], 'narr', ('WAIT.',), 'hard'),
    (K['wct'], K['l8'], 'narr', ('THAT', 'CHANGES', 'EVERYTHING.'), 'hard'),
    (K['l8'], K['cx'], 'narr', ('YOUR PERFECT', 'MOVE...'), 'wipe'),
    (K['cx'], K['thr1'], 'narr', ('JUST GOT', 'COMPLICATED.'), 'hard'),
    (K['thr1'], K['thr2'], 'narr', ('THREE', 'OPPONENTS.'), 'reveal'),
    (K['thr2'], K['share'], 'narr', ('MORE THAN', 'ONE MOVE', 'TO WATCH.'), 'reveal'),
    (K['share'], K['logo'], 'narr', ('SHARE IT WITH', 'YOUR FRIENDS'), 'reveal'),
]
TURNS = [(K['p1'], 'EARTH GREEN'), (K['p1a'], 'FIRE RED'), (K['p2'], 'AIR GREY'), (K['p2a'], 'WATER BLUE'),
         (K['p3'], 'AIR GREY')]
_land = lambda k0, k1, s0, s1, sl: K[k0] + (sl - s0) / (s1 - s0) * (K[k1] - K[k0])
LANDS = [_land('p1', 'p1a', 0.05, 1.9, 1.2), _land('p1a', 'p2', 3.3, 5.45, 5.1), _land('p2', 'p2a', 1.0, 3.0, 2.8),
         K['p2a'] + 0.6, _land('wct', 'l8', 3.0, 4.6, 4.0), K['hit'] - 0.05]
HOOK_HIT_T = 0.0
OPP_FLASH = [(K['thr1'], 'WATER BLUE'), (K['thr1'] + 0.4, 'AIR GREY'), (K['thr1'] + 0.8, 'EARTH GREEN')]
MONTAGE_HITS = [K['cx']]
LAND1_T = K['p1'] + 1.2 + (5.1 - 3.7) / (5.2 - 3.7) * 2.1          # FIRE RED piece lands (for the tick SFX)


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


@functools.lru_cache(maxsize=None)
def spark_mask():
    """Sparkles stay in the background: zero in the headline zone (y 180-560 at 1080), soft edges."""
    yy = np.arange(H, dtype=np.float32) / H * 1920
    m = 1 - np.clip(1 - np.maximum(180 - yy, yy - 560) / 80, 0, 1)
    return m[:, None, None].astype(np.float32)


def render(t):
    f = picture(t)
    for t0, army in TURNS + OPP_FLASH:                                 # army-colour edge glow: eased in and out
        if t0 <= t < t0 + 0.9:
            k = (t - t0) / 0.9
            env = math.sin(math.pi * min(1.0, k / 0.25) / 2) ** 2 * (1 - eio(seg(k, 0.25, 1.0)))
            f += edge_mask() * (np.array(ARMY[army], np.float32) / 255.0) * 0.28 * env
            if (t0, army) in TURNS:
                T.army_sweep(f, ARMY[army], eio(k), cy=HEAD_CY)
    for th in MONTAGE_HITS:
        if 0 <= t - th < 0.2:
            f = f * (1 + 0.35 * math.sin(math.pi * (t - th) / 0.2))
    hit = K['hit']
    if 0 <= t - hit < 0.4:                                             # capture: soft flash + short shake
        k = t - hit
        f = f * (1 + 0.35 * math.sin(math.pi * min(1.0, k / 0.25)))
        amp = P(12) * math.exp(-k / 0.12)
        M = np.float32([[1, 0, amp * math.sin(k * 70)], [0, 1, amp * 0.6 * math.cos(k * 53)]])
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    end = t >= K['card']
    if t < K['logo']:
        f *= top_bed()
    if not end:
        sp = np.zeros_like(f)
        EMB.draw(sp, t)
        BOKEH.draw(sp, t, ox=-t * 30)
        f += sp * spark_mask()
    f = C.grade(f, t, sat=1.12, leak=0.0 if end else 0.06, vig=0.3 if end else 0.62, grain=0.0 if end else 0.012,
                bloom=0.15 if end else 0.38)
    draw_text(f, t)
    for img, x, y, al in WHITE:                                       # name labels: after the grade, pure white
        C.place(f, img, x, y, 1.0, al)
    WHITE.clear()
    if end and not CLEAN:                                             # end-card label, fades in with the lockup
        lk_h = still('lockup.png').shape[0] * W * 0.98 / still('lockup.png').shape[1]
        C.place(f, label_img(), W / 2, P(HERO_Y) - lk_h / 2 - P(40), 1.0, 1.0)
    if t < 0.25:                                                      # opening fade in 0.25 s
        f *= eio(t / 0.25)
    return f
