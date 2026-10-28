"""Revision 3 timeline: follows the original script scene by scene, on the Mixkit "Epic Games" beat grid.

HOOK 0:00 | CREATE A PRIVATE ROOM 3.58 | CHOOSE YOUR GAME 8.15 | GET YOUR ROOM CODE 12.72 |
SHARE IT WITH YOUR FRIENDS 15.0 | THEY JOIN 17.29 | EVERYONE'S IN 20.72 | LET THE GAME BEGIN 23.0 (peak) |
PLAY WITH YOUR FRIENDS 25.86 | TRY IT NOW 28.72 | END CARD (uploaded outro) 29.86-34.0

No Hotseat anywhere: the private-room beat uses the real Private Room screen (Host or Join) and the
Create Room envelope screen, neither of which contains the Hotseat button.
"""
import functools
import math
import os

import cv2
import numpy as np
from PIL import Image

import common
import scenes as old
from common import (FPS, H, P, SC, W, band, clip, cover, dir_blur, draw_title, eio, eo3, flush_post, font, grade,
                    lerp, place, place_post, proj, remap, rotm, rrect_mask, seg, splat, still, text_mask, zoom_blur)
from pc import pc_frame

old.draw_title = lambda *a, **k: None   # captions are driven from this file's single schedule
MODE = os.environ.get('R3MODE', 'mobile')   # 'mobile' (phones + PC shots) or 'desktop' (no phones anywhere)
if MODE == 'desktop':
    old.HOLO_C = (1080, 1180)   # hologram rises above the monitor instead of above a phone
DURATION = 34.0
ROOT = common.ROOT
RED, GREEN, BLUE, WHITE, GOLD = old.RED, old.GREEN, old.BLUE, old.WHITE, old.GOLD

# ------------------------------------------------------------------ captions (one schedule, output clock)
CAPTIONS = [
    (('GOT THREE', 'FRIENDS?'), 1.29, 3.42, 900, 250, True),
    (('CREATE A', 'PRIVATE ROOM'), 3.62, 8.0, 640, 190, False),
    (('CHOOSE YOUR GAME',), 8.2, 12.58, 620, 150, False),
    (('GET YOUR', 'ROOM CODE'), 12.76, 14.86, 640, 190, False),
    (('SHARE IT WITH', 'YOUR FRIENDS'), 15.04, 17.15, 640, 175, False),
    (('THEY JOIN',), 17.33, 20.58, 600, 220, False),
    (("EVERYONE'S IN",), 20.76, 22.86, 640, 250, True),
    (('LET THE GAME', 'BEGIN'), 23.0, 25.72, 640, 210, False),
    (('PLAY WITH', 'YOUR FRIENDS'), 25.9, 28.6, 640, 190, False),
    (('TRY IT NOW',), 28.72, 29.78, 1920, 270, False),
]


@functools.lru_cache(maxsize=None)
def fit_size(lines, size, max_w=1820):
    _, w = text_mask(tuple(lines), size)
    return size * min(1.0, P(max_w) / max(w, 1))


def captions(f, t):
    for lines, t0, t1, cy, size, glass in CAPTIONS:
        draw_title(f, lines, t, t0, P(cy), t1=t1, size=fit_size(lines, size), glass=glass, shake=14 if glass else 10)


def label(text, size, font_name='Montserrat_wght_600.ttf', tracking=0.3, color=(0.92, 0.9, 0.86)):
    m, _ = text_mask((text,), size, font_name, tracking)
    sh = cv2.GaussianBlur(np.roll(m, int(P(4)), 0), (0, 0), P(8)) * 0.7
    a = np.clip(m + sh * (1 - m), 0, 1)
    rgb = np.where(m[..., None] > 0, np.array(color[::-1], np.float32), 0) * np.ones_like(m)[..., None]
    return np.dstack([rgb, a]).astype(np.float32)


# ------------------------------------------------------------------ HOOK
def k_hook(t):
    return old.s_hook(remap(t, [(0, 0), (1.29, 0.95), (2.15, 1.72), (2.43, 1.92), (3.58, 3.0)]))


# ------------------------------------------------------------------ CREATE A PRIVATE ROOM
PRIV_FOCUS = (495, 560)       # "Private Room" banner + Host or Join + Create Room / Join Room


def k_create(t):
    if t < 4.72:   # phone, tight on the real Private Room screen
        lt = t - 3.58
        scr = clip('rec_priv2', 60).at(remap(t, [(3.58, 0.1), (4.72, 1.3)]))
        pose = [0, 0, lerp(1.42, 1.22, eio(lt / 1.14)), lerp(11, 4, eio(lt / 1.14)), 5, lerp(-5, 0, eo3(lt / 0.3))]
        bg = old.arena('env_45.jpg', t, ox=pose[3] * 10)
        f, _, _ = old.shot(bg, scr, pose, PRIV_FOCUS, (W / 2, P(2200)), t, rim=(0.2, 0.45, 0.8))
        old.atmos(f, t, ox=pose[3] * 20)
        if t < 3.7:
            f = dir_blur(f, P(700) * (1 - eo3(seg(t, 3.58, 3.7))), 0, n=7)
        return f
    if t < 5.86:   # the same private-room framing on the real desktop monitor
        lt = t - 4.72
        ui = clip('rec_priv2', 60).at(remap(t, [(4.72, 1.3), (5.86, 2.5)]))
        f, _ = pc_frame('pc51602', 0.4 + lt, ui, focus_ui=PRIV_FOCUS, zoom=lerp(1.05, 1.32, eio(lt / 1.14)))
        if t < 4.84:
            f = zoom_blur(f, 0.1 * (1 - seg(t, 4.72, 4.84)))
        return f
    # envelope screen on the phone, push-in and slow-motion tap on Create Room (click on the 8.15 beat)
    t_old = remap(t, [(5.86, 8.45), (7.0, 10.3), (8.15, 11.67)])
    return old.s_create(t_old)


# ------------------------------------------------------------------ CHOOSE YOUR GAME
LOGOS = {
    'Quess': ('quess_logo.png', 1180, (960, 560)),
    'CheQuers': ('chequers_logo.png', 1640, (530, 560)),
    'BaQugammon': ('baqugammon_logo.png', 1640, (1390, 560)),
}
LOGO_T = [('Quess', 9.86), ('CheQuers', 11.0), ('BaQugammon', 12.15)]


@functools.lru_cache(maxsize=None)
def logo_img(name):
    fn, width, _ = LOGOS[name]
    im = np.asarray(Image.open(os.path.join(ROOT, 'assets2', fn)).convert('RGBA')).astype(np.float32) / 255.0
    ys, xs = np.nonzero(im[..., 3] > 0.02)
    im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = P(width) / im.shape[1]
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb = im[..., [2, 1, 0]]
    if name == 'BaQugammon':   # supplied as black artwork; shown in white like the other two (requested)
        rgb = np.full_like(rgb, 0.96)
    return np.dstack([rgb, im[..., 3]])


def frosted_card(f, cx, cy, w, h, a):
    """Light frosted-glass plate (behind the black BaQugammon artwork so it reads on a dark scene)."""
    x0, y0, x1, y1 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    roi = f[y0:y1, x0:x1]
    m = rrect_mask(x1 - x0, y1 - y0, int(P(48)))[..., None] * a
    frost = cv2.GaussianBlur(roi, (0, 0), P(30)) * 0.55 + np.array([0.80, 0.80, 0.80], np.float32) * 0.62
    ys = np.linspace(1.06, 0.9, y1 - y0, dtype=np.float32)[:, None, None]
    sh = cv2.GaussianBlur(np.roll(m[..., 0], int(P(14)), 0), (0, 0), P(24))[..., None] * 0.5
    roi[:] = roi * (1 - sh * (1 - m))
    roi[:] = roi * (1 - m) + np.clip(frost * ys, 0, 1) * m
    edge = np.clip(m[..., 0] - cv2.erode(m[..., 0], np.ones((3, 3), np.uint8), iterations=max(1, int(P(2)))), 0, 1)
    roi += edge[..., None] * 0.35


def k_choose(t):
    if t < 8.72:   # wide: the real room lit by the screen
        lt = t - 8.15
        img = clip('pc51622').at(3.0 + lt)
        f = cover(img, focus=(0.42, 0.5), zoom=lerp(1.0, 1.07, lt / 0.57))
        f *= 1.15
        if t < 8.35:
            f += 1.0 * (1 - eo3(seg(t, 8.15, 8.35)))
        return f
    if t < 9.86:   # real gaming setup, game selection on the monitor, push in
        lt = t - 8.72
        ui = clip('rec_select').at(0.2 + lt)
        f, _ = pc_frame('pc51602', 5.0 + lt, ui, zoom=lerp(0.92, 1.15, eio(lt / 1.14)), focus=(0.42, 0.45))
        if t < 8.82:
            f = zoom_blur(f, 0.12 * (1 - seg(t, 8.72, 8.82)))
        return f
    # phone close-ups that move panel to panel while each uploaded logo lands: Quess, CheQuers, BaQugammon
    src = remap(t, [(9.86, 2.05), (10.999, 2.95), (11.0, 3.1), (12.149, 3.55), (12.15, 1.0), (12.72, 1.57)])   # screen highlights the same game as the logo
    scr = clip('rec_select').at(src)
    pts = [LOGOS[n][2] for n, _ in LOGO_T]
    k1, k2 = eio(seg(t, 10.9, 11.1)), eio(seg(t, 12.05, 12.25))
    fx = lerp(lerp(pts[0][0], pts[1][0], k1), pts[2][0], k2)
    pose = [0, 0, 1.75, lerp(-8, 8, eio(seg(t, 9.86, 12.72))), 6, 0]
    bg = old.arena('env_10.5.jpg', t, ox=pose[3] * 10)
    f, _, _ = old.shot(bg, scr, pose, (fx, 520), (W / 2, P(1850)), t, rim=(0.25, 0.3, 0.7))
    whip = max(math.exp(-abs(t - 11.0) / 0.05), math.exp(-abs(t - 12.15) / 0.05))
    if whip > 0.05:
        f = dir_blur(f, P(500) * whip, 0, n=7)
    old.atmos(f, t, ox=pose[3] * 20, k=0.8)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    f *= 1 - 0.55 * np.clip((ys - 0.62) / 0.2, 0, 1)        # dark floor of the frame for the logos
    for i, (name, t0) in enumerate(LOGO_T):
        t1 = LOGO_T[i + 1][1] - 0.04 if i + 1 < len(LOGO_T) else 12.66
        if not (t0 <= t < t1 + 0.12):
            continue
        tau = t - t0
        e = 1 - math.exp(-tau / 0.07)
        a = min(1.0, tau / 0.1) * (1 - eio((t - t1) / 0.12) if t > t1 else 1)
        s = 1.08 - 0.08 * e
        img = logo_img(name)
        cy = P(3020)
        if tau < 0.18:
            img = cv2.blur(img, (1, max(1, int(P(30) * (1 - e))) * 2 + 1))
        place_post(img, P(1080), cy, s, a)
        if name == 'Quess' and tau > 0.15:
            place_post(label('4-PLAYER CHESS', 70), P(1080), P(3470), 1.0, a * eo3((tau - 0.15) / 0.2))
    return f


POST = common.POST


# ------------------------------------------------------------------ GET YOUR ROOM CODE
def k_code(t):
    return old.s_code(remap(t, [(12.72, 12.0), (13.29, 13.14), (14.43, 14.34), (15.0, 15.0)]))


# ------------------------------------------------------------------ SHARE IT WITH YOUR FRIENDS
CODE_C = (493, 807)


def k_share(t):
    if t < 15.86:   # real desktop close-up: the code and its Copy button
        lt = t - 15.0
        ui = clip('rec_code', 60).at(2.2 + lt)
        f, _ = pc_frame('pc51602', 1.2 + lt, ui, focus_ui=(560, 760), zoom=lerp(1.55, 1.85, eio(lt / 0.86)))
        if t < 15.12:
            f = zoom_blur(f, 0.12 * (1 - seg(t, 15.0, 15.12)))
        return f
    return old.s_share(remap(t, [(15.86, 15.3), (16.0, 15.38), (16.43, 16.02), (16.72, 16.32), (17.0, 16.62),
                                 (17.29, 17.0)]))


# ------------------------------------------------------------------ THEY JOIN (cinematic lobby)
SLOTS = [
    dict(name='AHMAD', army='RED ARMY', col=RED, still='mov_6.6.png', c=(2080, 1160), size=760, t=17.29),
    dict(name='JOHN GREEN', army='GREEN ARMY', col=GREEN, still='mov_4.0.png', c=(900, 1060), size=900, t=17.86),
    dict(name='JACKY', army='BLUE ARMY', col=BLUE, still='mov_4.0.png', c=(3250, 960), size=1000, t=19.01),
    dict(name='OPIE', army='WHITE ARMY', col=WHITE, still='mov_6.6.png', c=(3480, 1200), size=700, t=20.15),
]
CARD_W, CARD_H = 0.92, 1.18
CARD_POS = [(-0.5, -0.64), (0.5, -0.64), (-0.5, 0.64), (0.5, 0.64)]


@functools.lru_cache(maxsize=None)
def portrait(i):
    s = SLOTS[i]
    img = still(s['still']).astype(np.float32) / 255.0
    cx, cy = s['c']
    h = s['size'] // 2
    crop = img[max(0, cy - h):cy + h, max(0, cx - h):cx + h]
    d = int(P(560))
    return cv2.resize(crop, (d, d), interpolation=cv2.INTER_AREA)


@functools.lru_cache(maxsize=None)
def card_static():
    tw, th = int(P(1000)), int(P(1280))
    m = rrect_mask(tw, th, int(P(56)))
    d = int(P(560))
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
    r = np.sqrt((xx - d / 2) ** 2 + (yy - d / 2) ** 2)
    circ = np.clip((d / 2 - 1 - r) / max(1.0, P(3)), 0, 1)
    ring = np.clip(1 - np.abs(r - (d / 2 - P(4))) / P(4), 0, 1)
    return tw, th, m, circ, ring


def card_texture(i, t):
    """Foreground of one slot (portrait, name, army, accent) as RGBA; the glass is composited in frame space."""
    s = SLOTS[i]
    tw, th, m, circ, ring = card_static()
    k = eo3((t - s['t']) / 0.25) if t >= s['t'] else 0.0
    rgba = np.zeros((th, tw, 4), np.float32)
    d = circ.shape[0]
    px, py = (tw - d) // 2, int(P(120))
    pr = portrait(i)
    lum = pr.mean(2, keepdims=True)
    empty = lum * 0.06
    face = empty * (1 - k) + pr * k
    col = np.array(s['col'], np.float32)
    ringc = 0.22 * (1 - k) + col * k
    rgb = face * (1 - ring[..., None]) + ringc * ring[..., None]
    a = np.maximum(circ, ring) * (0.85 + 0.15 * k)
    rgba[py:py + d, px:px + d] = np.dstack([rgb, a])
    if k > 0:   # name sweeps in, army label under it, accent line in the army colour
        nm, _ = text_mask((s['name'],), 96, 'Cinzel_wght_900.ttf', 0.06)
        sc = min(1.0, (tw * 0.86) / nm.shape[1])
        nm = cv2.resize(nm, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        wipe = eo3((t - s['t'] - 0.05) / 0.35)
        cols = (np.arange(nm.shape[1]) <= wipe * nm.shape[1]).astype(np.float32)
        nm = nm * cols[None, :]
        ny = int(th * 0.66)
        nx = (tw - nm.shape[1]) // 2
        hh, ww = nm.shape
        reg = rgba[ny:ny + hh, nx:nx + ww]
        reg[..., :3] = reg[..., :3] * (1 - nm[..., None]) + 0.97 * nm[..., None]
        reg[..., 3] = np.maximum(reg[..., 3], nm)
        am, _ = text_mask((s['army'],), 46, 'Montserrat_wght_600.ttf', 0.3)
        am = am * eo3((t - s['t'] - 0.2) / 0.25)
        ay = ny + hh + int(P(6))
        ax = (tw - am.shape[1]) // 2
        hh2, ww2 = am.shape
        if ay + hh2 < th:
            reg2 = rgba[ay:ay + hh2, ax:ax + ww2]
            reg2[..., :3] = reg2[..., :3] * (1 - am[..., None]) + 0.72 * am[..., None]
            reg2[..., 3] = np.maximum(reg2[..., 3], am)
        lw = int(tw * 0.5 * eo3((t - s['t']) / 0.3))
        ly = th - int(P(70))
        rgba[ly:ly + max(1, int(P(6))), (tw - lw) // 2:(tw + lw) // 2] = np.concatenate([col, [1.0]])
    return rgba, k


def k_join(t):
    lt = t - 17.29
    joined = sum(1 for s in SLOTS if t >= s['t'])
    mt = 2.6 + lt * 0.9
    img = clip('mov').at(mt - 1.0)
    bg = band(img, focus=(0.5, 0.5), zoom=2.9 + 0.1 * lt, bg_dim=0.3)
    small = cv2.resize(bg, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (W, H)) * (0.55 + 0.07 * joined)
    f = bg.copy()
    frost_src = cv2.resize(cv2.GaussianBlur(small, (0, 0), 7), (W, H))
    # camera drifts through the lobby and leans toward each player as they join
    last = max(i for i, s in enumerate(SLOTS) if t >= s['t'])
    lean = eio(seg(t, SLOTS[last]['t'], SLOTS[last]['t'] + 0.5))
    tx = CARD_POS[last][0] * 0.12 * lean
    pose_yaw = lerp(-13, 9, eio(lt / 3.43))
    pose_pitch = lerp(9, 3, eio(lt / 3.43))
    z = lerp(3.25, 2.85, eio(lt / 3.43)) - 0.12 * (joined == 4) * eo3(seg(t, 20.15, 20.72))
    R = rotm(pose_yaw, pose_pitch, lerp(-2, 1, lt / 3.43))
    T = np.array([-tx, 0.12, z])
    tw, th, m, _, _ = card_static()
    for i, (cx, cy) in enumerate(CARD_POS):
        bob = 0.015 * math.sin(t * 1.4 + i * 1.7)
        loc = np.array([[cx - CARD_W / 2, cy - CARD_H / 2 + bob, 0], [cx + CARD_W / 2, cy - CARD_H / 2 + bob, 0],
                        [cx + CARD_W / 2, cy + CARD_H / 2 + bob, 0], [cx - CARD_W / 2, cy + CARD_H / 2 + bob, 0]])
        q = proj((R @ loc.T).T + T)
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), q)
        a = cv2.warpPerspective(m, Hm, (W, H))[..., None]
        sh = cv2.GaussianBlur(np.roll(a[..., 0], int(P(30)), 0), (0, 0), P(40))[..., None] * 0.6
        f *= 1 - sh * (1 - a)
        tex, k = card_texture(i, t)
        col = np.array(SLOTS[i]['col'], np.float32)
        # frosted glass: blurred scene behind, lifted, lightly tinted by the army once joined
        glass = frost_src * 0.8 + 0.12 + col * 0.08 * k
        f[:] = f * (1 - a * 0.9) + glass * a * 0.9
        edge = np.clip(a[..., 0] - cv2.erode(a[..., 0], np.ones((3, 3), np.uint8), iterations=max(1, int(P(2)))), 0, 1)
        f += edge[..., None] * (0.28 + 0.4 * k * np.array([1, 1, 1], np.float32))
        fg = cv2.warpPerspective(tex, Hm, (W, H))
        fa = fg[..., 3:4]
        f[:] = f * (1 - fa) + fg[..., :3] * fa
        if t >= SLOTS[i]['t'] and i > 0:
            u = (t - SLOTS[i]['t']) / 0.45
            if u < 1:  # light sweep across the glass + a soft burst of the army colour behind the slot
                xs = np.arange(W, dtype=np.float32)
                band_ = np.exp(-((xs - lerp(q[:, 0].min(), q[:, 0].max(), u)) / P(70)) ** 2)[None, :, None]
                f += a * band_ * 0.5 * (1 - u)
                c = q.mean(0)
                lay = np.zeros((H // 8, W // 8, 3), np.float32)
                cv2.circle(lay, (int(c[0] / 8), int(c[1] / 8)), int(P(520) / 8), tuple(float(v) for v in col), -1)
                f += cv2.resize(cv2.GaussianBlur(lay, (0, 0), P(160) / 8), (W, H)) * 0.9 * (1 - u) * (1 - a)
    if t >= 20.15:   # all four seats lit: the room surges
        u = seg(t, 20.15, 20.72)
        f += 0.25 * math.exp(-(t - 20.15) / 0.12) + 0.6 * eio(seg(t, 20.5, 20.72)) ** 2
    old.atmos(f, t, k=0.6 + 0.2 * joined)
    return f


# ------------------------------------------------------------------ EVERYONE'S IN / LET THE GAME BEGIN
def k_all_in(t):
    return old.s_all_in(remap(t, [(20.72, 21.0), (23.0, 23.0)]))


def k_begin(t):
    if 24.72 <= t < 25.29:   # one gameplay cut on the real desktop, over the gamer's shoulder
        lt = t - 24.72
        ui = clip('gp1b').at(0.6 + lt * 2.0)
        f, _ = pc_frame('pc51614', 1.6 + lt, ui, focus=(0.4, 0.42), zoom=1.12 * (1 + 0.1 * (1 - eo3(lt / 0.2))))
        if t < 24.84:
            f += 0.5 * (1 - seg(t, 24.72, 24.84))
        return f
    return old.s_begin(remap(t, [(23.0, 23.0), (24.15, 23.85), (24.72, 24.6), (25.29, 25.35), (25.86, 26.0)]))


# ------------------------------------------------------------------ PLAY WITH YOUR FRIENDS / TRY IT NOW
def k_friends(t):
    f, warm = old.s_friends(remap(t, [(25.86, 26.0), (27.0, 27.4), (27.57, 27.9), (28.72, 29.3), (29.86, 30.0)]))
    return f, warm


# ------------------------------------------------------------------ END CARD (uploaded outro)
def k_end(t):
    img = clip('outro').at(t - 29.86)
    f = cover(img, zoom=1.0)
    place_post(label('QUESS - 4 PLAYER CHESS', 96, 'Cinzel_wght_900.ttf', 0.08, (0.97, 0.96, 0.94)),
               P(1080), P(2970), 1.0, eo3(seg(t, 32.43, 32.75)))
    place_post(label('AN INTERNET CONNECTION IS NEEDED TO PLAY.', 40, 'Montserrat_wght_500.ttf', 0.18,
                     (0.78, 0.78, 0.8)), P(1080), P(3190), 1.0, eo3(seg(t, 32.72, 33.05)))
    return f


# ------------------------------------------------------------------ assembly
SHAKES = [(1.29, 18), (2.15, 10), (2.43, 10), (3.62, 10), (8.15, 18), (9.86, 8), (11.0, 8), (12.15, 8),
          (12.76, 10), (15.04, 10), (16.0, 14), (17.33, 10), (17.86, 10), (19.01, 10), (20.15, 16), (20.76, 12),
          (23.0, 36), (24.15, 18), (24.72, 18), (25.29, 18), (25.9, 10), (27.0, 20), (28.72, 40), (29.86, 12)]


def shake_at(t):
    dx = dy = 0.0
    for tc, amp in SHAKES:
        if 0 <= t - tc < 0.45:
            k = t - tc
            a = P(amp) * math.exp(-k / 0.09)
            dx += a * math.sin(k * 87 + tc)
            dy += a * 0.7 * math.cos(k * 71 + tc * 3)
    return dx, dy


def render(t):
    warm = 0.0
    if t < 3.58:
        f = k_hook(t)
    elif t < 8.15:
        f = k_create(t)
    elif t < 12.72:
        f = k_choose(t)
    elif t < 15.0:
        f = k_code(t)
    elif t < 17.29:
        f = k_share(t)
    elif t < 20.72:
        f = k_join(t)
    elif t < 23.0:
        f = k_all_in(t)
    elif t < 25.86:
        f = k_begin(t)
    elif t < 29.86:
        f, warm = k_friends(t)
    else:
        f = k_end(t)
    return _finish(f, t, warm)


def _finish(f, t, warm):
    if 3.48 <= t < 3.7:   # hook push resolves into a flash
        f += 1.1 * math.exp(-abs(t - 3.58) / 0.05)
    dx, dy = shake_at(t)
    if abs(dx) + abs(dy) > 0.3:
        M = np.array([[1.02, 0, -0.01 * W + dx], [0, 1.02, -0.01 * H + dy]], np.float32)
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    captions(f, t)
    if t >= 29.86:   # the uploaded outro is finished artwork: shown as delivered
        f = np.clip(f, 0, 1)
    else:
        f = grade(f, t, warm=warm, leak=0.12 if warm else 0.09)
    return flush_post(f)


# ================================================================== DESKTOP VERSION (no phones anywhere)
_M = dict(create=k_create, choose=k_choose, share=k_share, friends=k_friends)   # phone-version shots
BTN_C = (493, 833)


def d_create(t):
    if t < 4.72:   # tight on the real Private Room screen, a different camera angle of the real setup
        lt = t - 3.58
        ui = clip('rec_priv2', 60).at(remap(t, [(3.58, 0.1), (4.72, 1.3)]))
        f, _ = pc_frame('pc51602', 9.0 + lt, ui, focus_ui=PRIV_FOCUS, zoom=lerp(1.2, 1.45, eio(lt / 1.14)))
        if t < 3.7:
            f = dir_blur(f, P(700) * (1 - eo3(seg(t, 3.58, 3.7))), 0, n=7)
        return f
    if t < 5.86:
        return _M['create'](t)
    # envelope screen: push in on the real monitor toward Create Room, slow-motion click on the 8.15 beat
    u = eio(seg(t, 5.86, 8.05))
    abs_t = remap(t, [(5.86, 37.15), (7.0, 39.0), (8.15, 39.675)])
    ui = clip('rec_env', 60).at(abs_t - 36.0)
    pulse = 0.55 + 0.35 * math.sin((t - 5.86) * 7.0)
    ui = old.screen_glow(ui.copy(), old.ENVELOPE, GOLD, 0.35 * pulse, width=26)
    ui = old.screen_glow(ui, old.SEAL, (0.2, 0.45, 1.0), 0.45 * pulse, width=30, inner=0.25)
    fu = (lerp(560, BTN_C[0], u), lerp(600, BTN_C[1], u))
    f, to_out = pc_frame('pc51602', 3.0 + (t - 5.86), ui, focus_ui=fu, zoom=lerp(1.1, 2.0, u))
    old.atmos(f, t, k=0.4)
    return f


def click_burst(f, t, xy):
    if t < 8.15 or t > 8.6:
        return f
    k = math.exp(-(t - 8.15) / 0.12)
    lay = np.zeros((H // 4, W // 4, 3), np.float32)
    cv2.circle(lay, (int(xy[0] / 4), int(xy[1] / 4)), int(P(160) / 4), (0.8, 0.9, 1.0), -1)
    cv2.ellipse(lay, (int(xy[0] / 4), int(xy[1] / 4)), (int(W * 0.7 / 4), max(1, int(P(14) / 4))), 0, 0, 360,
                (1.0, 0.85, 0.6), -1)
    lay = cv2.GaussianBlur(lay, (0, 0), P(18) / 4) + cv2.GaussianBlur(lay, (0, 0), P(90) / 4)
    return f + cv2.resize(lay, (W, H)) * 2.2 * k + 0.55 * k


def d_choose(t):
    if t < 9.86:
        f = _M['choose'](t)
        if t < 8.6:   # the click flash carries over the cut
            f = click_burst(f, t, (W / 2, P(1920)))
        return f
    src = remap(t, [(9.86, 2.05), (10.999, 2.95), (11.0, 3.1), (12.149, 3.55), (12.15, 1.0), (12.72, 1.57)])   # screen highlights the same game as the logo
    ui = clip('rec_select').at(src)
    pts = [LOGOS[n][2] for n, _ in LOGO_T]
    k1, k2 = eio(seg(t, 10.9, 11.1)), eio(seg(t, 12.05, 12.25))
    fx = lerp(lerp(pts[0][0], pts[1][0], k1), pts[2][0], k2)
    f, _ = pc_frame('pc51602', 6.6 + (t - 9.86), ui, focus_ui=(fx, 470), zoom=1.55)
    whip = max(math.exp(-abs(t - 11.0) / 0.05), math.exp(-abs(t - 12.15) / 0.05))
    if whip > 0.05:
        f = dir_blur(f, P(500) * whip, 0, n=7)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    f *= 1 - 0.6 * np.clip((ys - 0.62) / 0.2, 0, 1)
    for i, (name, t0) in enumerate(LOGO_T):
        t1 = LOGO_T[i + 1][1] - 0.04 if i + 1 < len(LOGO_T) else 12.66
        if not (t0 <= t < t1 + 0.12):
            continue
        tau = t - t0
        e = 1 - math.exp(-tau / 0.07)
        a = min(1.0, tau / 0.1) * (1 - eio((t - t1) / 0.12) if t > t1 else 1)
        img = logo_img(name)
        if tau < 0.18:
            img = cv2.blur(img, (1, max(1, int(P(30) * (1 - e))) * 2 + 1))
        place_post(img, P(1080), P(3020), 1.08 - 0.08 * e, a)
        if name == 'Quess' and tau > 0.15:
            place_post(label('4-PLAYER CHESS', 70), P(1080), P(3470), 1.0, a * eo3((tau - 0.15) / 0.2))
    return f


def d_code(t):
    """The room code lifts off the real monitor as gold particles and settles as a hologram above it."""
    t_old = remap(t, [(12.72, 12.0), (13.29, 13.14), (14.43, 14.34), (15.0, 15.0)])
    src = remap(t_old, [(12.0, 0.33), (12.6, 0.93), (15.0, 2.13)])
    ui = clip('rec_code', 60).at(src)
    lt = t - 12.72
    f, to_out = pc_frame('pc51602', 10.0 + lt * 0.8, ui, focus_ui=(800, 520), zoom=lerp(1.02, 1.12, eio(lt / 2.28)),
                         dy=P(380))
    if t < 12.84:
        f += 0.6 * (1 - seg(t, 12.72, 12.84))
    A = old.holo_assets()
    cb = old.CODE_BOX
    box = to_out([(cb[0], cb[1]), (cb[2], cb[1]), (cb[2], cb[3]), (cb[0], cb[3])])
    hx, hy = P(old.HOLO_C[0]), P(old.HOLO_C[1])
    kb = eo3(seg(t_old, 12.6, 12.9))
    if kb > 0:
        lay = np.zeros((H // 4, W // 4, 3), np.float32)
        hw = A['w'] / 2
        poly = np.array([box[0], box[1], (hx + hw, hy + A['h'] * 0.3), (hx - hw, hy + A['h'] * 0.3)]) / 4
        cv2.fillConvexPoly(lay, poly.astype(np.int32), GOLD, cv2.LINE_AA)
        f += cv2.resize(cv2.GaussianBlur(lay, (0, 0), P(30) / 4), (W, H)) * 0.16 * kb
    ci = A['ci']
    u = np.clip((t_old - (old.CHAR_T0 + ci * old.CHAR_DT + A['jit'] * 0.05)) / old.CHAR_FLY, 0, 1)
    live = (u > 0) & (u < 1.0)
    if live.any():
        s = A['starts']
        a, b, c, dd = box
        sx = (a[0] * (1 - s[:, 0]) + b[0] * s[:, 0]) * (1 - s[:, 1]) + (dd[0] * (1 - s[:, 0]) + c[0] * s[:, 0]) * s[:, 1]
        sy = (a[1] * (1 - s[:, 0]) + b[1] * s[:, 0]) * (1 - s[:, 1]) + (dd[1] * (1 - s[:, 0]) + c[1] * s[:, 0]) * s[:, 1]
        e = 1 - (1 - u) ** 3
        x = sx + (A['tx'] - sx) * e + np.sin(u * math.pi) * A['rng_dir'][:, 0] * P(60)
        y = sy + (A['ty'] - sy) * e - np.sin(u * math.pi) * P(140)
        inten = (0.9 * (1 - u * 0.6))[:, None] * np.array(GOLD, np.float32)
        splat(f, x[live], y[live], inten[live])
    old.draw_holo_crisp(f, old.col_alpha(t_old))
    return f


def d_share(t):
    if t < 15.86:
        return _M['share'](t)
    return old.s_share(remap(t, [(15.86, 15.45), (16.43, 16.02), (16.72, 16.32), (17.0, 16.62), (17.29, 17.0)]))


D_CUTS = [   # gameplay on the real PC: start, end, clip, clip time at start, speed, stock clip, stock time, focus, zoom
    (23.0, 24.15, 'tra', 0.45, 1.5, 'pc51602', 7.2, (0.40, 0.42), 1.25),
    (24.15, 24.72, 'gp1a', 0.4, 2.0, 'pc51614', 0.2, (0.42, 0.40), 1.08),
    (24.72, 25.29, 'gp1b', 0.6, 2.0, 'pc51614', 1.6, (0.40, 0.42), 1.12),
    (25.29, 25.86, 'gp1b', 1.8, 1.6, 'pc51602', 11.0, (0.45, 0.42), 1.2),
]


def d_cut(t, c):
    t0, t1, name, a0, sp, pcn, pt0, foc, z = c
    ui = clip(name).at(a0 + (t - t0) * sp)
    f, _ = pc_frame(pcn, pt0 + (t - t0), ui, focus=foc, zoom=z * (1 + 0.1 * (1 - eo3((t - t0) / 0.2))))
    return f


def d_begin(t):
    for i, c in enumerate(D_CUTS):
        if c[0] <= t < c[1] or (i == len(D_CUTS) - 1 and t >= c[0]):
            f = d_cut(t, c)
            if i == 0 and t > 23.95:   # the capture smoke wipes into the next shot
                nxt = d_cut(D_CUTS[1][0] + 0.001, D_CUTS[1])
                k = eio(seg(t, 23.95, 24.15))
                m = np.clip((f.mean(2) - (0.85 - 0.8 * k)) * 4, 0, 1)
                m = np.clip(cv2.GaussianBlur(m, (0, 0), P(30))[..., None] * (0.5 + 1.5 * k), 0, 1)
                f = f * (1 - m) + nxt * m + m * (1 - m) * 1.2
            break
    if t < 23.25:
        f += 1.0 * (1 - eo3(seg(t, 23.0, 23.25)))
    for tc in (24.15, 24.72, 25.29):
        if tc <= t < tc + 0.1:
            f += 0.45 * (1 - (t - tc) / 0.1)
    old.atmos(f, t, k=0.5)
    return f


def d_friends(t):
    if 27.0 <= t < 27.57:
        ui = clip('gp1b').at(1.2 + (t - 27.0) * 1.4)
        f, _ = pc_frame('pc51602', 8.2 + (t - 27.0), ui, focus=(0.42, 0.42), zoom=1.3 * (1 + 0.12 * (1 - eo3((t - 27.0) / 0.15))))
        if t < 27.1:
            f += 0.55 * (1 - (t - 27.0) / 0.1)
        return f, 0.0
    if t >= 28.72:   # TRY IT NOW on the real setup, final gameplay hit
        lt = t - 28.72
        ui = clip('tra').at(1.05 + lt * 0.7)
        f, _ = pc_frame('pc51614', 3.4 + lt, ui, focus=(0.42, 0.42), zoom=1.15 * (1 + 0.15 * (1 - eo3(lt / 0.15))))
        if t < 28.84:
            f += 0.55 * (1 - lt / 0.12)
        if t > 29.66:
            f *= 1 - eio(seg(t, 29.66, 29.86)) * 0.85
        return f, 0.0
    return _M['friends'](t)


if MODE == 'desktop':
    k_create, k_choose, k_code, k_share, k_begin, k_friends = d_create, d_choose, d_code, d_share, d_begin, d_friends
