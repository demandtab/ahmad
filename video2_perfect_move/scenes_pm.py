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
}
K = {k: S(v) for k, v in N.items()}
DURATION = round((K['outro'] + 4.8) * FPS) / FPS
STOP = K['hit'] - 0.25                                   # 0.25 s stop-down before the hit


def shots():
    """(start, end, draw(local_t, t) -> frame, transition-in seconds)."""
    p1, p1m, p1a, p2, p2a, p3, hit = K['p1'], K['p1m'], K['p1a'], K['p2'], K['p2a'], K['p3'], K['hit']
    out = []

    # 1 HOOK: EARTH GREEN character studying the real board, slow 3D push-in with a slight roll
    def hook(lt, t):
        k = lt / p1
        return frame_from('hook', 0.6 + lt * 0.85, (0.47, 0.56), lerp(1.04, 1.24, eio(k)), angle=lerp(0.0, -1.2, k))
    out.append((0.0, p1, hook, 0.0))

    # 2 FIRE RED, BEFORE: whole board, piece selected (real legal-move dots)
    d_b = p1m - p1
    def p1_before(lt, t):
        return frame_from('g1', 0.25 + lt * (2.6 / d_b), (0.40, 0.52), lerp(1.25, 1.31, lt / d_b), band=True)
    out.append((p1, p1m, p1_before, 0.20))

    # 2 FIRE RED, MOVE: punch-in, slow motion on the fire-trail move, spotlight follows the piece
    d_m = p1a - p1m
    def p1_move(lt, t):
        k = lt / d_m
        src = 2.85 + (lt * 0.9 if lt < 1.0 else 0.9 + (lt - 1.0) * (1.85 / max(d_m - 1.0, 0.1)))   # ramp into 0.6x
        foc = (lerp(0.37, 0.41, k), lerp(0.50, 0.60, k))
        z = lerp(1.05, 1.32, eio(k))
        f = frame_from('g1', src, foc, z)
        px = (lerp(0.35, 0.45, eio(seg(src, 3.9, 5.1))), lerp(0.47, 0.70, eio(seg(src, 3.9, 5.1))))
        cx, cy = to_out('g1', foc, z, px)
        return spotlight(f, cx, cy, P(560), 0.55)
    out.append((p1m, p1a, p1_move, 0.20))

    # 3 MOVE PLAYED: whole board after the move, steady hold then punch-in on the landed piece
    d_a = p2 - p1a
    def p1_after(lt, t):
        k = seg(lt, d_a * 0.55, d_a)
        src = min(5.6 + lt * 0.45, 6.3)
        return frame_from('g1', src, (lerp(0.40, 0.40, k), lerp(0.52, 0.56, k)), lerp(1.25, 1.55, eio(k)), band=True)
    out.append((p1a, p2, p1_after, 0.20))

    # 4 EARTH GREEN: wider drift, selection then slow-motion move with spotlight
    d2 = p2a - p2
    def p2_move(lt, t):
        k = lt / d2
        src = 0.05 + lt * 0.18 if lt < 0.8 else 0.05 + 0.144 + (lt - 0.8) * (1.0 / max(d2 - 0.8, 0.1))
        foc = (lerp(0.38, 0.43, k), 0.46)
        f = frame_from('tra', src, foc, 1.12, angle=lerp(1.5, 0.0, k))
        px = (lerp(0.40, 0.44, seg(src, 0.25, 1.2)), lerp(0.46, 0.36, seg(src, 0.25, 1.2)))
        cx, cy = to_out('tra', foc, 1.12, px)
        return spotlight(f, cx, cy, P(520), 0.5)
    out.append((p2, p2a, p2_move, 0.20))

    # 5 "Ok... still fine.": the changed board, tighter whole-board framing, uneasy slow drift
    d5 = p3 - p2a
    def p2_after(lt, t):
        k = lt / d5
        return frame_from('tra', 1.25 + lt * 0.95, (lerp(0.47, 0.44, k), 0.50), lerp(1.40, 1.48, k), band=True,
                          angle=lerp(0.0, -1.0, k))
    out.append((p2a, p3, p2_after, 0.20))

    # 6 AIR GREY: low dutch angle, speed ramp into the capture, then a 0.25 s stop-down hold
    d6 = hit - p3
    def p3_in(lt, t):
        k = lt / d6
        src = 3.0 + min(lt, d6 - 0.25) * (2.25 / max(d6 - 0.25, 0.1))
        f = frame_from('tra', src, (0.38, lerp(0.50, 0.44, k)), lerp(1.0, 1.18, ei(k)), angle=lerp(-3.0, -1.0, k))
        if lt > d6 - 0.25:
            f *= 0.78                                      # the stop-down: picture holds and dips
        return f
    out.append((p3, hit, p3_in, 0.20))

    # 7 IMPACT: the real capture burst, sharp impact framing, no blur (shake added in render)
    d7 = K['react'] - hit
    def impact(lt, t):
        return frame_from('tra', 5.32 + lt * 0.95, (0.36, 0.42), lerp(1.28, 1.36, lt / d7))
    out.append((hit, K['react'], impact, 0.0))

    # 7 cut-back to the FIRE RED player (no text)
    def react(lt, t):
        return frame_from('react', lt * 0.35, (0.40, 0.47), lerp(1.0, 1.14, eio(lt / (K['l8'] - K['react']))))   # face + glowing eyes
    out.append((K['react'], K['l8'], react, 0.12))

    # 8 chaos montage, every cut a hit
    def native(lt, t):
        f = seq('c096').at(0.2 + lt)
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        f = C.cover(f, zoom=lerp(1.08, 1.0, lt / 1.4))
        return sharpen(f) * 0.95 + 0.025
    out.append((K['l8'], K['c097'], native, 0.06))
    out.append((K['c097'], K['fcp'], lambda lt, t: frame_from('c097', 0.1 + lt, (0.56, 0.52), lerp(1.0, 1.1, lt)), 0.06))
    out.append((K['fcp'], K['burn'], lambda lt, t: frame_from('fcp', 0.1 + lt, (0.5, 0.55), lerp(1.12, 1.0, lt)), 0.06))
    out.append((K['burn'], K['qc'], lambda lt, t: frame_from('burn', 0.4 + lt * 1.4, (0.5, 0.5), 1.15), 0.06))
    d_qc = K['end'] - K['qc']
    out.append((K['qc'], K['end'], lambda lt, t: frame_from('qc', 0.3 + lt, (0.5, 0.55),
                                                             lerp(1.65, 1.0, eo3(lt / d_qc))), 0.06))

    # END: four players on thrones, then the real outro lifted as one untouched layer
    d_e = K['outro'] - K['end']
    out.append((K['end'], K['outro'], lambda lt, t: frame_from('thr', 0.4 + lt, (0.5, 0.55),
                                                               lerp(1.12, 1.0, lt / d_e)), 0.20))

    def outro(lt, t):
        f = seq('outro').at(0.0 + lt)
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        M = np.float32([[1, 0, 0], [0, 1, -P(460) * eio(seg(lt, 1.2, 2.0))]])   # lift as the lockup forms
        return cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
    out.append((K['outro'], DURATION, outro, 0.30))
    return out


def ei(x):
    return C.ei3(x)


SHOTS = shots()

# ------------------------------------------------------------------ text events
TEXT = [  # (t0, t1, kind, lines, fx)
    (S(0.45), S(1.75), 'narr', ('WHEN YOU', 'FINALLY FIND'), 'embers'),
    (S(1.75), K['p1'] - 0.05, 'narr', ('THE PERFECT', 'MOVE...'), 'reveal'),
    (S(3.25), K['p1m'] - 0.05, 'thought', ('I know exactly', 'what to do.'), 'reveal'),     # on the BEFORE panel
    (S(7.5), K['p2'] - 0.05, 'narr', ('MOVE PLAYED.',), 'glow'),
    (K['p2'], S(10.5), 'narr', ('THEN PLAYER 2', 'MOVES.'), 'reveal'),
    (S(11.15), K['p3'] - 0.05, 'thought', ('Ok... still fine.',), 'reveal'),
    (K['p3'], STOP, 'narr', ('THEN PLAYER 3', 'MOVES.'), 'reveal'),
    (K['hit'], S(K['hit'] + 0.86) - 1 / FPS, 'thought', ('Wait.',), 'hard'),                # on the capture panel
    (S(K['hit'] + 0.86), K['react'] - 1 / FPS, 'thought', ('That changes', 'everything.'), 'hard'),
    (K['l8'], K['fcp'] - 0.05, 'narr', ('YOUR PERFECT', 'MOVE...'), 'wipe'),
    (K['fcp'], K['end'] - 0.05, 'narr', ('JUST GOT', 'COMPLICATED.'), 'reveal'),
    (S(K['end'] + 0.2), S(27.3) - 0.05, 'narr', ('THREE', 'OPPONENTS.'), 'reveal'),
    (S(27.3), K['outro'] - 0.05, 'narr', ('MORE THAN', 'ONE MOVE', 'TO WATCH.'), 'reveal'),
    (S(K['outro'] + 1.75), DURATION + 1, 'narr', ('PLAY QUESS.',), 'embers'),
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
        out_k = clamp01((t - (t1 - 0.16)) / 0.16)       # leaves before the next block lands: never > max lines
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
    f *= T.text_bed(0.45)
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
