"""Quess trailer, LOCKED recipe ("exact like Quess 19th .1"): every layer from the client's beat table.

Layers are placed in 1080x1920 coordinates (pos=x,y; w = fraction of the 1080 frame width) exactly as the
RECIPE states, rendered at any scale (SC). All images are real assets (src2/lock, cut from the Drive recordings
and the real outro). z order: 0 stars + gold sparkles, 2 heroes, 4 code/logo, 5 headline + underline (nothing
is ever drawn over text).

  SC=0.166667 python3 render_lock.py video out.mp4     # 360p
"""
import functools
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'trailer'))
import common as C  # noqa: E402
from common import W, H, FPS, eo3, eio, seg, lerp, clamp01  # noqa: E402
import type2 as T  # noqa: E402

LOCK = os.path.join(C.ROOT, 'src2', 'lock')
CLEAN = os.environ.get('CLEAN') == '1'
DURATION = 35.0
U = W / 1080.0                              # 1080-space unit -> output pixels
GOLD = np.array((74, 162, 201), np.float32) / 255.0     # #C9A24A (BGR)
IVORY = np.array((0.95, 0.96, 0.97), np.float32)
SAFE_TOP, SAFE_W = 220, 800


# ------------------------------------------------------------------ assets
@functools.lru_cache(maxsize=None)
def asset(name):
    im = cv2.imread(os.path.join(LOCK, name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    if im.shape[2] == 3:
        im = np.dstack([im, np.ones(im.shape[:2], np.float32)])
    return im


@functools.lru_cache(maxsize=64)
def asset_w(name, wpx):
    """Asset pre-scaled to an integer output width (cached: placement then only crops/blends)."""
    im = asset(name)
    h = max(1, int(round(im.shape[0] * wpx / im.shape[1])))
    out = cv2.resize(im, (wpx, h), interpolation=cv2.INTER_AREA if wpx < im.shape[1] else cv2.INTER_CUBIC)
    rgb = out[..., :3]
    out[..., :3] = np.clip(rgb + (rgb - cv2.GaussianBlur(rgb, (0, 0), 1.0)) * 0.35, 0, 1)   # light sharpening
    return out


def layer(f, name, x, y, w, scale=1.0, alpha=1.0, tilt=0.0, blend='over', cam=None):
    """Draw asset centred at (x, y) [1080 space] with width w (fraction of 1080) * scale, optional yaw tilt
    (degrees, real perspective) and camera zoom cam=(zoom, cx, cy) applied to the layer."""
    if cam:
        z, ccx, ccy = cam
        x, y, scale = ccx + (x - ccx) * z, ccy + (y - ccy) * z, scale * z
    wpx = max(2, int(round(w * 1080 * U * scale)))
    if wpx > 6 * W:
        wpx = 6 * W
    im = asset_w(name, wpx)
    h, wd = im.shape[:2]
    cx, cy = x * U, y * U
    if abs(tilt) > 0.05:
        th = math.radians(tilt)
        dist = wd * 2.2
        pts = []
        for px_, py_ in ((-wd / 2, -h / 2), (wd / 2, -h / 2), (wd / 2, h / 2), (-wd / 2, h / 2)):
            X, Z = px_ * math.cos(th), px_ * math.sin(th) + dist
            pts.append((cx + X * dist / Z, cy + py_ * dist / Z))
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [wd, 0], [wd, h], [0, h]]), np.float32(pts))
        warped = cv2.warpPerspective(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = warped[..., 3:4] * alpha
        f[:] = f * (1 - a) + warped[..., :3] * a
        return
    x0, y0 = int(round(cx - wd / 2)), int(round(cy - h / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + wd), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    src = im[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = src[..., 3:4] * alpha
    roi = f[Y0:Y1, X0:X1]
    if blend == 'screen':                      # real crops on a dark starfield: no box edge
        roi[:] = roi + (1 - roi) * src[..., :3] * a
    else:
        roi[:] = roi * (1 - a) + src[..., :3] * a


# ------------------------------------------------------------------ type (Cinzel Black, one look everywhere)
def wid(text, px, tr=0.06):
    fnt = C.font('Cinzel_wght_900.ttf', px)
    return sum(fnt.getlength(ch) for ch in text) + px * tr * (len(text) - 1)


def wrap(text, px):
    lines, cur = [], ''
    for wd in text.split():
        c = (cur + ' ' + wd).strip()
        if wid(c, px) <= SAFE_W * U:
            cur = c
        else:
            lines.append(cur)
            cur = wd
    lines.append(cur)
    return lines


@functools.lru_cache(maxsize=None)
def text_block(text, size, rule=False, lines=None):
    """White Cinzel 900 text (size in px at 1080), centred lines, soft shadow; optional gold underline
    (w=0.30 of the frame, 4 px, directly under). Returns (rgba, letters_top_row, letters_bottom_row)."""
    px = size * U
    lines = lines or ((text,) if wid(text, px) <= SAFE_W * U else tuple(wrap(text, px)))
    tm, tw, marg = T._mask(tuple(lines), size * 2, gap=0.30)     # T works in 4K units (x2)
    hh, ww = tm.shape
    rw, rh, gap = int(0.30 * 1080 * U), max(1, int(round(4 * U))), int(px * 0.32)
    ww2 = max(ww, rw + 8)
    padx = (ww2 - ww) // 2
    tm = np.pad(tm, ((0, (gap + rh + int(20 * U)) if rule else 0), (padx, ww2 - ww - padx)))
    rows = np.nonzero(tm.max(1) > 0.3)[0]
    top, bot = int(rows[0]), int(rows[-1])
    am = np.zeros_like(tm)
    if rule:
        y = bot + gap
        xs = np.arange(ww2, dtype=np.float32)
        am[y:y + rh] = (np.clip(1 - np.abs(xs - ww2 / 2) / (rw / 2), 0, 1) ** 0.3)[None, :]
    sil = np.maximum(tm, am)
    sh = np.roll(sil, max(1, int(3 * U)), 0)
    shadow = np.clip(cv2.GaussianBlur(sh, (0, 0), max(0.6, 3 * U)) * 0.8 + cv2.GaussianBlur(sh, (0, 0), max(1, 14 * U)) * 0.5, 0, 0.85)
    rgb = np.where(sil[..., None] > 0, (tm[..., None] * IVORY + am[..., None] * GOLD) / np.maximum(sil[..., None], 1e-4), 0)
    return np.dstack([rgb, np.clip(sil + shadow * (1 - sil), 0, 1)]).astype(np.float32), top, bot


def draw_text(f, text, size, x, y, scale=1.0, alpha=1.0, rule=False, lines=None, keep_top=True):
    img, top, bot = text_block(text, size, rule, lines)
    h = img.shape[0]
    cy = y * U - ((top + bot) / 2 - h / 2)           # letters' centre on y
    if keep_top and cy - h / 2 + top < (SAFE_TOP + 20) * U:   # never into the top UI zone
        cy = (SAFE_TOP + 20) * U - top + h / 2
    C.place(f, img, x * U, cy, scale, alpha)


def headline_size(text):
    return 100 if wid(text, 100 * U) <= SAFE_W * U else 88


# ------------------------------------------------------------------ the locked beat table
HEADLINES = [   # (t0, t1, text): continuous 00:00-00:28, out=cut, in=scale 1.08->1.0 over 0.25 s
    (0.00, 3.00, 'WHEN YOU FINALLY FIND THE PERFECT MOVE...'),
    (3.00, 7.00, 'CREATE A PRIVATE ROOM'),
    (7.00, 11.00, 'CHOOSE YOUR GAME'),
    (11.00, 15.00, 'GET YOUR ROOM CODE'),
    (15.00, 19.00, 'THEY JOIN'),
    (19.00, 21.50, 'LET THE GAME BEGIN'),
    (21.50, 24.00, 'THEN PLAYER 2 MOVES. THEN PLAYER 3 MOVES.'),
    (24.00, 26.00, 'YOUR PERFECT MOVE JUST GOT COMPLICATED.'),
    (26.00, 28.00, 'THREE OPPONENTS.'),
]
CARDS = [  # (file, x, y, label name, label army, t)
    ('card_teller_fire_red.png', 330, 820, 'TELLER', 'FIRE RED', 15.30),
    ('card_john_green_earth_green.png', 750, 820, 'JOHN GREEN', 'EARTH GREEN', 15.90),
    ('card_jacky_water_blue.png', 330, 1250, 'JACKY', 'WATER BLUE', 16.50),
    ('card_opie_air_grey.png', 750, 1250, 'OPIE', 'AIR GREY', 17.10),
]
ARMY = {'WATER BLUE': (240, 150, 30), 'AIR GREY': (215, 215, 215), 'EARTH GREEN': (60, 210, 70)}
EDGE = [(26.00, 'WATER BLUE'), (26.40, 'AIR GREY'), (26.80, 'EARTH GREEN')]

SPARK = C.Embers(31, 110, size=(2, 5), speed=(40, 120), color=(0.40, 0.72, 1.0), down=3, blur=1.2, gain=0.55)


@functools.lru_cache(maxsize=1)
def stars():
    img = cv2.imread(os.path.join(C.ROOT, 'src2', 'stills', 'bg_stars.png')).astype(np.float32) / 255.0
    return C.cover(img, zoom=1.2)


@functools.lru_cache(maxsize=None)
def edge_mask():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    e = np.minimum.reduce([xx, yy, W - xx, H - yy]) / (120 * U)
    return np.exp(-e ** 2)[..., None]


def pop(k):
    """easeOutBack 0 -> 1 (pop-in)."""
    k = clamp01(k)
    c = 1.70158
    return 1 + (c + 1) * (k - 1) ** 3 + c * (k - 1) ** 2


def picture(t):
    end = t >= 30.0
    f = stars().copy() * (0.85 if not end else 0.8)
    if not end:
        SPARK.draw(f, t)                                         # z=0 only
    if t < 3.0:                                                  # 1: board low angle, push-in 1.00 -> 1.08
        layer(f, 'board_lowangle.png', 540, 960, 3.16, cam=(lerp(1.0, 1.08, t / 3.0), 540, 960))
    elif t < 7.0:                                                # 2: Private Room, tilt 18 -> 0, zoom from 4.20
        lt = t - 3.0
        z = lerp(1.0, 1.10, eio(seg(t, 4.20, 7.0)))
        layer(f, 'private_room.png', 540, 980, 0.94, tilt=18 * (1 - eo3(lt / 1.2)), cam=(z, 540, 980))
    elif t < 11.0:                                               # 3: Choose Game, tilt -16 -> 0, push from 8.20
        lt = t - 7.0
        z = lerp(1.0, 1.05, eio(seg(t, 8.20, 11.0)))
        layer(f, 'choose_game.png', 540, 940, 0.94, tilt=-16 * (1 - eo3(lt / 1.2)), cam=(z, 540, 940))
    elif t < 15.0:                                               # 4: room code panel, fade 0.3, camera into the code
        lt = t - 11.0
        z = lerp(1.0, 1.12, eio(lt / 4.0))
        layer(f, 'room_code_panel.png', 540, 1180, 0.80, alpha=clamp01(lt / 0.3), cam=(z, 540, 820))
    elif t < 19.0:                                               # 5: they join, cards pop 0.25 s, 0.6 s apart
        for name, x, y, *_rest, tc in CARDS:
            if t >= tc:
                layer(f, name, x, y, 0.36, scale=pop((t - tc) / 0.25))
    elif t < 21.5:                                               # 6: capture board, fast push 1.00 -> 1.15
        layer(f, 'board_capture.png', 918, 960, 2.84, cam=(lerp(1.0, 1.15, eo3((t - 19.0) / 2.5)), 540, 960))
    elif t < 24.0:                                               # 7: low angle, new crop, push 1.00 -> 1.12
        layer(f, 'board_lowangle.png', 1180, 1060, 4.4, cam=(lerp(1.0, 1.12, eio((t - 21.5) / 2.5)), 540, 960))
    elif t < 28.0:                                               # 8: ONE top-down board, static
        layer(f, 'board_topdown.png', 540, 960, 3.16)
    elif t < 30.0:                                               # logo: scale 0.6 -> 1.0 over 0.8 s, fade out 0.2
        lt = t - 28.0
        layer(f, 'quess_q_logo.png', 540, 900, 0.55, scale=lerp(0.6, 1.0, eo3(lt / 0.8)),
              alpha=clamp01(lt / 0.12) * (1 - clamp01((lt - 1.8) / 0.2)))
    else:                                                        # end card: fade 0.3, then static
        a = clamp01((t - 30.0) / 0.3)
        for name, y, w in (('company_logo.png', 780, 0.70), ('available_on.png', 990, 0.28),
                           ('store_badges.png', 1090, 0.76), ('url.png', 1210, 0.42)):
            layer(f, name, 540, y, w, alpha=a, blend='screen')
    # capture burst shape 00:20.75-00:21.00 (#FF6A2A, w=0.30 at 540,922): flash 0.08, fade 0.17
    if 20.75 <= t < 21.0:
        k = 1.0 if t < 20.83 else 1 - (t - 20.83) / 0.17
        yy, xx = np.ogrid[0:H, 0:W]
        r = np.sqrt((xx - 540 * U) ** 2 + (yy - 922 * U) ** 2) / (0.15 * 1080 * U)
        f += (np.exp(-r ** 2 * 1.5) * k * 1.2)[..., None] * np.array((42, 106, 255), np.float32) / 255
    for t0, army in EDGE:                                        # edge flashes: flash 0.05, fade 0.3
        if t0 <= t < t0 + 0.35:
            k = 1.0 if t < t0 + 0.05 else 1 - (t - t0 - 0.05) / 0.3
            f += edge_mask() * np.array(ARMY[army], np.float32) / 255 * 0.8 * k
    return f


def render(t):
    f = picture(t)
    end = t >= 30.0
    f = C.grade(f, t, sat=1.10, leak=0.0 if end or t >= 28 else 0.05, vig=0.3 if end else 0.55,
                grain=0.0 if end else 0.010, bloom=0.1 if end else 0.32)
    if t < 0.25:
        f *= eio(t / 0.25)                                       # in=fade:0.25
    if CLEAN:
        return f
    for t0, t1, text in HEADLINES:                               # z=5, out=cut
        if t0 <= t < t1:
            k = eo3((t - t0) / 0.25)
            draw_text(f, text, headline_size(text), 540, 330, scale=lerp(1.08, 1.0, k), alpha=clamp01((t - t0) / 0.06),
                      rule=True)
    if 8.50 <= t < 11.0:                                         # bottom label: the selected game
        draw_text(f, 'CheQuers', 100, 540, 1400, scale=lerp(1.3, 1.0, eo3((t - 8.5) / 0.3)), keep_top=False)
    if 11.0 <= t < 15.0:                                         # room code hero text (zooms with the camera)
        z = lerp(1.0, 1.12, eio((t - 11.0) / 4.0))
        draw_text(f, 'CH-EAAF79', 122, 540, 820, scale=lerp(1.3, 1.0, eo3((t - 11.0) / 0.3)) * z, keep_top=False)
    if 15.0 <= t < 19.0:                                         # card labels pop with their cards
        for _n, x, y, name, army, tc in CARDS:
            if t >= tc:
                draw_text(f, None, 34, x, y + 215, scale=pop((t - tc) / 0.25), lines=(name, army), keep_top=False)
    if end:
        draw_text(f, 'QUESS - 4 PLAYER CHESS', 53, 540, 560, alpha=clamp01((t - 30.0) / 0.3), keep_top=False)
    return f
