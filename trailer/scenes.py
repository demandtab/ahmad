"""Shot-by-shot timeline of the Quess vertical trailer (34.0 s @ 30 fps).

VARIANT=v1  phone mockups everywhere, except the Hotseat shot which plays on a gaming PC (Hotseat is PC-only)
VARIANT=v2  every UI / gameplay shot plays on the gaming PC setup, no phones anywhere

Scene code is written on the original edit's clock ("old" time). `render(t)` maps the
output clock onto it with BEAT_MAP so every cut lands on the beat grid of Mixkit
"Epic Games" (105 BPM, peak drop at 0:23.0).
"""
import functools
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import (FPS, flush_post, place_post, H, P, SC, W, Embers, arena, band, clip, device, device_shot, desk_shot, dir_blur,
                    draw_title, ei3, eio, eo3, font, grade, lerp, monitor, place, remap, screen_glow, seg,
                    splat, still, text_img, zoom_blur)

VARIANT = os.environ.get('VARIANT', 'v1')
DURATION = 34.0

# army colours (BGR)
RED = (0.16, 0.18, 1.0)
GREEN = (0.30, 1.0, 0.25)
BLUE = (1.0, 0.55, 0.12)
WHITE = (1.0, 0.97, 0.92)
GOLD = (0.30, 0.74, 1.0)
RGB_ROOM = (0.95, 0.25, 0.70)

DUST = Embers(3, 90, size=(1.5, 3.5), speed=(20, 80), color=(0.6, 0.65, 0.7), gain=0.35)
EMB = Embers(1, 130, size=(2, 5), speed=(90, 300), color=(0.18, 0.5, 1.0), gain=0.9)
BOKEH = Embers(2, 14, size=(20, 48), speed=(260, 620), color=(0.2, 0.55, 1.0), down=6, blur=5, gain=0.3)

# output time -> original-edit time; every left-hand value sits on the Epic Games beat grid
BEAT_MAP = [(0, 0), (1.29, 0.95), (2.15, 1.72), (2.43, 1.92), (3.58, 3.0),
            (4.72, 4.2), (5.29, 5.2), (5.86, 6.0), (7.01, 7.0),
            (8.15, 7.8), (8.72, 8.45), (10.43, 10.3), (11.58, 11.67), (11.86, 12.0),
            (12.72, 13.14), (13.86, 14.34), (15.0, 15.0),
            (15.57, 15.38), (16.15, 16.02), (16.72, 16.32), (17.29, 16.62), (18.44, 18.0),
            (18.87, 18.9), (19.01, 19.05), (19.58, 19.6), (20.15, 20.15), (20.72, 21.0),
            (23.0, 23.0), (24.15, 23.85), (24.72, 24.6), (25.29, 25.35), (25.86, 26.0),
            (27.0, 27.4), (27.57, 27.9), (29.29, 29.3), (29.86, 30.0),
            (31.0, 31.2), (31.57, 31.6), (34.0, 34.0)]


def new2old(t):
    return remap(t, BEAT_MAP)


def old2new(t):
    return remap(t, [(b, a) for a, b in BEAT_MAP])


def atmos(frame, t, ox=0.0, k=1.0):
    DUST.draw(frame, t, ox * 0.4, gain=k)
    EMB.draw(frame, t, ox, gain=k)
    BOKEH.draw(frame, t, ox * 1.8, gain=k)


def shot(bg, scr, pose, aim_src, aim_tgt, t, desk=False, light=RGB_ROOM, rim=None, glow=0.6, kb=True):
    """Put a real recording on the phone (default) or on the gaming PC monitor."""
    if desk:
        m = monitor()
        pose = list(pose)
        pose[2] *= 1.1
        pose[4] = pose[4] * 0.3
        pose = m.aim(pose, aim_src, (aim_tgt[0], aim_tgt[1] - P(320)))
        f, Hm = desk_shot(bg * 0.45, scr, pose, t, light=light, glow=glow * 0.8, kb=kb)
        return f, Hm, m
    d = device()
    pose = d.aim(pose, aim_src, aim_tgt)
    f, Hm = device_shot(bg, scr, pose, glow=glow, rim_light=rim)
    return f, Hm, d


def fit_screen(img):
    """Gameplay letterboxed (never cropped) into the 1920x1006 screen space."""
    h, w = img.shape[:2]
    from common import SRC_H, SRC_W
    cw, ch = int(P(SRC_W)) if SC < 1 else SRC_W, int(P(SRC_H)) if SC < 1 else SRC_H
    s = min(cw / w, ch / h)
    r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((ch, cw, 3), np.float32)
    y0, x0 = (ch - r.shape[0]) // 2, (cw - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def gameplay_on_pc(img, t, t0, yaw, light=RGB_ROOM):
    z = 2.35 * (1 - 0.1 * (1 - eo3((t - t0) / 0.2))) * (1 - 0.03 * (t - t0))
    pose = [0, 0, z, yaw, 3, 0]
    bg = arena('env_45.jpg', t, ox=yaw * 10)
    f, _, _ = shot(bg, fit_screen(img), pose, (960, 503), (W / 2, P(2050)), t, desk=True, light=light, glow=0.7)
    return f


# ============================================================ 0:00 HOOK
def s_hook(t):
    img = clip('env_hook').at(t + 1.0)
    if t < 1.45:
        zw = lerp(4.15, 4.45, t / 1.45)
    elif t < 2.2:
        zw = lerp(4.45, 1.0, eio(seg(t, 1.45, 2.2)))
    else:
        zw = lerp(1.0, 5.2, ei3(seg(t, 2.2, 3.0)))
    f = band(img, focus=(0.485, 0.5), zoom=zw, bg_dim=0.3)
    f *= lerp(0.0, 1.0, eo3(t / 0.35)) * (0.75 + 0.25 * seg(t, 1.45, 2.0))
    for tc, x, col in ((1.72, 0.12, WHITE), (1.92, 0.88, GREEN)):
        k = math.exp(-max(0.0, t - tc) / 0.22) * (t >= tc - 0.04)
        if k > 0.01:
            lay = np.zeros((H // 8, W // 8, 3), np.float32)
            cv2.ellipse(lay, (int(W * x / 8), int(H * 0.5 / 8)), (int(W * 0.22 / 8), int(H * 0.2 / 8)), 0, 0, 360, col, -1)
            lay = cv2.GaussianBlur(lay, (0, 0), W / 8 * 0.08)
            f += cv2.resize(lay, (W, H)) * 0.9 * k
    atmos(f, t, ox=-40 * t)
    draw_title(f, ('GOT THREE', 'FRIENDS?'), t, 0.95, P(860), t1=2.12, size=270)
    if t > 2.2:
        f = zoom_blur(f, 0.16 * ei3(seg(t, 2.2, 3.0)))
    return f


# ============================================================ 0:03 CHOOSE YOUR GAME
def ignite(t, t0, nxt=None, peak=1.6, hold=0.55):
    if t < t0:
        return 0.0
    k = peak * math.exp(-(t - t0) / 0.25) + hold * eo3((t - t0) / 0.1)
    if nxt is not None and t > nxt:
        k *= lerp(1.0, 0.35, eo3((t - nxt) / 0.2))
    return k


def game_light(t):
    """Environment light shifts to each army colour as its game is picked (light only, never on text)."""
    col = np.array((0.25, 0.25, 0.6), np.float32)
    for t0, c in ((4.2, GREEN), (5.2, BLUE), (6.0, RED)):
        if t >= t0:
            col = np.array(c, np.float32) * (1 + 0.8 * math.exp(-(t - t0) / 0.25))
    return col


def s_choose(t):
    src = min(t - 2.8, 3.55)
    scr = clip('rec_select').at(src)
    lt = t - 3.0
    yaw = lerp(-22, -9, eio(lt / 4.0))
    pose = [0, 0, lerp(2.85, 2.65, eio(lt / 4.0)), yaw, 7, 2.5]
    bg = arena('env_10.5.jpg', t, ox=yaw * 12)
    light = game_light(t)
    desk = VARIANT == 'v2'
    f, Hm, _ = shot(bg, scr, pose, (960, 503), (W / 2, P(2000)), t, desk=desk, light=light * 0.8,
                    rim=tuple(light * 0.5), glow=0.6)
    if not desk:  # coloured side light in the arena
        lay = np.zeros((H // 8, W // 8, 3), np.float32)
        cv2.circle(lay, (W // 8, int(H * 0.45 / 8)), int(W * 0.5 / 8), tuple(float(v) for v in light * 0.6), -1)
        f += cv2.resize(cv2.GaussianBlur(lay, (0, 0), W / 8 * 0.25), (W, H)) * 0.35
    atmos(f, t, ox=yaw * 25, k=0.6 if desk else 1.0)
    draw_title(f, ('CHOOSE YOUR GAME',), t, 3.05, P(700), size=210)
    # exact product names, clean white type, no glow
    draw_title(f, ('CheQuers',), t, 4.2, P(3060), t1=5.12, size=230, accent=False)
    draw_title(f, ('BaQugammon',), t, 5.2, P(3060), t1=5.92, size=230, accent=False)
    draw_title(f, ('Quess',), t, 6.0, P(3010), size=330, accent=False)
    draw_title(f, ('4-PLAYER CHESS',), t, 6.1, P(3300), size=120)
    if t > 6.82:  # whip out
        f = dir_blur(f, P(900) * ei3(seg(t, 6.82, 7.0)), 0, n=7)
    return f


# ============================================================ 0:07 HOTSEAT (PC) + CREATE A PRIVATE ROOM
ENVELOPE = (320, 446, 665, 711)
SEAL = (455, 548, 525, 618)
BTN_C = (493, 833)
HOTSEAT_C = (960, 650)
CLICK_T = 11.67


def s_create(t):
    desk = VARIANT == 'v2'
    if t < 7.8:  # Hotseat is a PC-only mode: always on the desktop setup, highlighted and readable
        lt = t - 7.0
        scr = clip('rec_hot', 60).at(remap(t, [(7.0, 0.2), (7.8, 1.0)]))
        pose = [0, 0, lerp(2.5, 2.05, eio(lt / 0.8)), lerp(-10, -4, lt / 0.8), 4, lerp(-6, 0, eo3(lt / 0.2))]
        sp = (lerp(960, HOTSEAT_C[0], eio(lt / 0.8)), lerp(503, HOTSEAT_C[1], eio(lt / 0.8)))
        bg = arena('env_45.jpg', t, ox=pose[3] * 10)
        f, Hm, _ = shot(bg, scr, pose, sp, (W / 2, P(2050)), t, desk=True, light=(0.2, 0.22, 1.0))
        atmos(f, t, k=0.5)
        draw_title(f, ('HOTSEAT MODE', 'ON PC'), t, 7.05, P(640), t1=7.74, size=210)
        if t < 7.12:
            f = dir_blur(f, P(900) * (1 - eo3(seg(t, 7.0, 7.12))), 0, n=7)
        return f
    if t < 8.45:
        scr = clip('rec_private').at(remap(t, [(7.8, 12.95), (8.45, 14.42)]) - 12.9)
    elif t < CLICK_T:
        abs_t = remap(t, [(8.45, 37.15), (10.3, 39.0), (CLICK_T, 39.675)])
        scr = clip('rec_env', 60).at(abs_t - 36.0)
    else:
        scr = clip('rec_code', 60).at(t - CLICK_T + 0.01)
    if 8.45 <= t < CLICK_T:  # the envelope art glows (art only, never the text or buttons)
        pulse = 0.55 + 0.35 * math.sin((t - 8.45) * 7.0)
        scr = screen_glow(scr.copy(), ENVELOPE, GOLD, 0.35 * pulse * eo3((t - 8.45) / 0.3), width=26)
        scr = screen_glow(scr, SEAL, (0.2, 0.45, 1.0), 0.45 * pulse, width=30, inner=0.25)
    if t < 8.45:
        lt = t - 7.8
        pose = [0, 0, 2.6, lerp(12, 8, lt / 0.65), 4, lerp(6, 0, eo3(lt / 0.15))]
        sp, tgt = (960, 503), (W / 2, P(2000))
    else:
        u = eio(seg(t, 8.45, 11.55))
        z = lerp(2.6, 1.5, u)
        if t > CLICK_T:
            z = lerp(1.5, 1.85, eo3(seg(t, CLICK_T, 12.0)))
        pose = [0, 0, z, lerp(8, 2, u), lerp(4, 2, u), 0]
        sp = (lerp(960, BTN_C[0], u), lerp(503, BTN_C[1], u))
        tgt = (W / 2, lerp(P(2000), P(2250), u))
    bg = arena('env_45.jpg', t, ox=pose[3] * 10)
    f, Hm, dev = shot(bg, scr, pose, sp, tgt, t, desk=desk, rim=(0.2, 0.45, 0.8))
    atmos(f, t, ox=pose[3] * 20, k=0.6 if desk else 1.0)
    draw_title(f, ('CREATE A', 'PRIVATE ROOM'), t, 7.85, P(700), t1=11.5, size=230)
    if t < 7.9:  # whip from the PC shot
        f = dir_blur(f, P(900) * (1 - eo3(seg(t, 7.8, 7.9))), 0, n=7)
    if t >= CLICK_T - 0.02:  # light burst from the clicked button
        bx, by = dev.map_screen(Hm, [BTN_C])[0]
        k = math.exp(-(t - CLICK_T) / 0.12)
        lay = np.zeros((H // 4, W // 4, 3), np.float32)
        cv2.circle(lay, (int(bx / 4), int(by / 4)), int(P(160) / 4), (0.8, 0.9, 1.0), -1)
        cv2.ellipse(lay, (int(bx / 4), int(by / 4)), (int(W * 0.7 / 4), max(1, int(P(14) / 4))), 0, 0, 360,
                    (1.0, 0.85, 0.6), -1)
        lay = cv2.GaussianBlur(lay, (0, 0), P(18) / 4) + cv2.GaussianBlur(lay, (0, 0), P(90) / 4)
        f += cv2.resize(lay, (W, H)) * 2.2 * k
        f += 0.55 * k
    return f


# ============================================================ 0:12 GET YOUR ROOM CODE (+ share)
CODE = 'CH-EAAF79'
CODE_BOX = (332, 776, 655, 838)
HOLO_C = (1080, 1500)
CHAR_T0, CHAR_DT, CHAR_FLY = 12.72, 0.15, 0.42


@functools.lru_cache(maxsize=None)
def holo_assets():
    f = font('Cinzel-Bold.ttf', P(300))
    b = f.getbbox(CODE)
    pad = int(P(40))
    w, h = b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).text((pad - b[0], pad - b[1]), CODE, font=f, fill=255)
    mask = np.asarray(m).astype(np.float32) / 255.0
    edges = [pad - b[0] + f.getlength(CODE[:i]) for i in range(len(CODE) + 1)]
    # settled code: clean white type with a soft dark shadow (no glow, no gradient)
    sh = cv2.GaussianBlur(np.roll(mask, int(P(8)), 0), (0, 0), P(14)) * 0.7
    alpha = np.clip(mask + sh * (1 - mask), 0, 1)
    rgb = np.where(mask[..., None] > 0, 0.96, 0.0) * np.ones((h, w, 3), np.float32)
    img = np.dstack([rgb, alpha]).astype(np.float32)
    rng = np.random.default_rng(11)
    ys, xs = np.nonzero(mask > 0.5)
    per = max(60, int(700 * SC * SC))
    pts = []
    for i in range(len(CODE)):
        idx = np.nonzero((xs >= edges[i]) & (xs < edges[i + 1]))[0]
        if len(idx) == 0:
            continue
        for j in rng.choice(idx, min(per, len(idx)), replace=False):
            pts.append((i, xs[j], ys[j]))
    pts = np.array(pts, np.float32)
    ox, oy = P(HOLO_C[0]) - w / 2, P(HOLO_C[1]) - h / 2
    return dict(img=img, w=w, h=h, edges=edges, ci=pts[:, 0].astype(int), tx=pts[:, 1] + ox, ty=pts[:, 2] + oy,
                starts=rng.uniform(0, 1, (len(pts), 2)).astype(np.float32),
                jit=rng.uniform(0, 1, len(pts)).astype(np.float32),
                rng_dir=rng.normal(0, 1, (len(pts), 2)).astype(np.float32),
                rnd=rng.uniform(0, 1, len(pts)).astype(np.float32))


def char_arrivals():
    return [CHAR_T0 + i * CHAR_DT + CHAR_FLY for i in range(len(CODE))]


def draw_holo_crisp(f, alpha_cols, a=1.0):
    A = holo_assets()
    img = A['img'].copy()
    img[..., 3] *= alpha_cols[None, :]
    place_post(img, P(HOLO_C[0]), P(HOLO_C[1]), 1.0, a)


def col_alpha(t):
    A = holo_assets()
    cols = np.zeros(A['w'], np.float32)
    for i, ta in enumerate(char_arrivals()):
        x0, x1 = int(A['edges'][i]), int(math.ceil(A['edges'][i + 1]))
        cols[max(0, x0):x1] = eo3((t - ta + 0.05) / 0.18)
    return cols


def code_pose(t):
    return [0, 0, 2.7, lerp(-9, 4, eio((t - 12.0) / 3.0)), 12, 0]


def s_code(t):
    desk = VARIANT == 'v2'
    src = remap(t, [(12.0, 0.33), (12.6, 0.93), (15.0, 2.13)])
    scr = clip('rec_code', 60).at(src)
    pose = code_pose(t)
    bg = arena('env_45.jpg', t, ox=pose[3] * 10, dim=0.27)
    f, Hm, dev = shot(bg, scr, pose, (960, 503), (W / 2, P(2600)), t, desk=desk, rim=(0.25, 0.6, 0.9), glow=0.55,
                      kb=False)
    atmos(f, t, ox=pose[3] * 20, k=0.6 if desk else 0.8)
    if t < 12.12:
        f += 0.6 * (1 - seg(t, 12.0, 12.12))
    A = holo_assets()
    box = dev.map_screen(Hm, [(CODE_BOX[0], CODE_BOX[1]), (CODE_BOX[2], CODE_BOX[1]),
                              (CODE_BOX[2], CODE_BOX[3]), (CODE_BOX[0], CODE_BOX[3])])
    kb = eo3(seg(t, 12.6, 12.9))
    if kb > 0:  # projection beam from the code up to the hologram
        lay = np.zeros((H // 4, W // 4, 3), np.float32)
        hw = A['w'] / 2
        poly = np.array([box[0], box[1], (P(HOLO_C[0]) + hw, P(HOLO_C[1]) + A['h'] * 0.3),
                         (P(HOLO_C[0]) - hw, P(HOLO_C[1]) + A['h'] * 0.3)]) / 4
        cv2.fillConvexPoly(lay, poly.astype(np.int32), GOLD, cv2.LINE_AA)
        f += cv2.resize(cv2.GaussianBlur(lay, (0, 0), P(30) / 4), (W, H)) * 0.16 * kb
    ci = A['ci']
    u = np.clip((t - (CHAR_T0 + ci * CHAR_DT + A['jit'] * 0.05)) / CHAR_FLY, 0, 1)
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
    draw_holo_crisp(f, col_alpha(t))
    draw_title(f, ('GET YOUR', 'ROOM CODE'), t, 12.05, P(640), t1=14.9, size=230)
    return f


AVATARS = [
    dict(still='mov_4.0.png', c=(900, 1060), size=900, col=GREEN, pos=(520, 2560), t=16.02),
    dict(still='mov_4.0.png', c=(3250, 960), size=1000, col=BLUE, pos=(1080, 2320), t=16.32),
    dict(still='mov_6.6.png', c=(3480, 1200), size=700, col=WHITE, pos=(1640, 2560), t=16.62),
]
AV_R = 250
BURST_T = 15.38


@functools.lru_cache(maxsize=None)
def avatar_imgs(i):
    a = AVATARS[i]
    img = still(a['still']).astype(np.float32) / 255.0
    cx, cy = a['c']
    s = a['size'] // 2
    crop = img[max(0, cy - s):cy + s, max(0, cx - s):cx + s]
    d = int(P(AV_R * 2))
    crop = cv2.resize(crop, (d, d), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
    r = np.sqrt((xx - d / 2) ** 2 + (yy - d / 2) ** 2)
    m = np.clip((d / 2 - 1 - r) / max(1.0, P(3)), 0, 1)
    ring = np.clip(1 - np.abs(r - (d / 2 - P(10))) / P(8), 0, 1)
    lum = crop.mean(2, keepdims=True)
    dim = (lum * 0.6 + crop * 0.4) * 0.38
    col = np.array(a['col'], np.float32)
    lit = np.dstack([crop * (1 - ring[..., None]) + col * ring[..., None], m])
    unlit = np.dstack([dim * (1 - ring[..., None]) + 0.25 * ring[..., None], m])
    ring_img = np.dstack([np.ones((d, d, 3), np.float32) * col, cv2.GaussianBlur(ring, (0, 0), P(14))])
    return lit, unlit, ring_img


def s_share(t):
    desk = VARIANT == 'v2'
    pose = code_pose(t)
    drop = ei3(seg(t, 15.0, 15.45)) * P(2400)
    bg = arena('env_45.jpg', t, ox=pose[3] * 10, dim=0.27)
    if t < 15.45:
        scr = clip('rec_code', 60).at(2.13 + (t - 15.0))
        f, _, _ = shot(bg, scr, pose, (960, 503), (W / 2, P(2600) + drop), t, desk=desk, rim=(0.25, 0.6, 0.9),
                       glow=0.55, kb=False)
        f = dir_blur(f, 0, P(260) * ei3(seg(t, 15.0, 15.45)))
    else:
        f = bg.copy()
        ys = np.arange(H, dtype=np.float32)
        f *= (1 - 0.5 * np.clip((ys - P(2850)) / P(500), 0, 1))[:, None, None]
    atmos(f, t, ox=-20 * (t - 15), k=0.9)
    A = holo_assets()
    if t < BURST_T:
        draw_holo_crisp(f, np.ones(A['w'], np.float32))
    else:
        rnd = A['rnd']
        x_rel = (A['tx'] - P(HOLO_C[0])) / (A['w'] / 2)
        k = np.clip(((x_rel + 1) / 2 * 3).astype(int), 0, 2)
        ta = np.array([a['t'] for a in AVATARS])[k] - 0.05 + rnd * 0.08
        td = BURST_T + rnd * 0.06
        u = np.clip((t - td) / (ta - td), 0, 1)
        live = u < 1
        if live.any():
            p0x, p0y = A['tx'], A['ty']
            p2x = np.array([P(a['pos'][0]) for a in AVATARS])[k]
            p2y = np.array([P(a['pos'][1]) for a in AVATARS])[k]
            dirs = A['rng_dir'] / (np.linalg.norm(A['rng_dir'], axis=1, keepdims=True) + 1e-6)
            c1x = p0x + dirs[:, 0] * P(500) * (0.5 + rnd)
            c1y = p0y + dirs[:, 1] * P(420) * (0.5 + rnd) - P(200)
            for lag, gain in ((0.0, 1.0), (0.05, 0.45), (0.1, 0.2)):
                uu = np.clip(u - lag, 0, 1)
                e = uu * uu * (3 - 2 * uu)
                x = (1 - e) ** 2 * p0x + 2 * (1 - e) * e * c1x + e ** 2 * p2x
                y = (1 - e) ** 2 * p0y + 2 * (1 - e) * e * c1y + e ** 2 * p2y
                cols = np.repeat(np.array(GOLD, np.float32)[None] * (gain * 0.9), len(x), 0)
                splat(f, x[live], y[live], cols[live])
        if t < BURST_T + 0.15:
            f += 0.5 * (1 - seg(t, BURST_T, BURST_T + 0.15))
    for i, a in enumerate(AVATARS):
        lit, unlit, ring = avatar_imgs(i)
        k = eo3((t - a['t']) / 0.18)
        ax, ay = P(a['pos'][0]), P(a['pos'][1] + 14 * math.sin(t * 1.6 + i * 2))
        appear = eo3(seg(t, 15.05 + i * 0.06, 15.35 + i * 0.06))
        if appear <= 0:
            continue
        sc = lerp(0.6, 1.0, appear) * (1 + 0.12 * math.exp(-max(0, t - a['t']) / 0.1) * (t > a['t']))
        place(f, unlit, ax, ay, sc, appear * (1 - k))
        place(f, lit, ax, ay, sc, appear * k)
        place(f, ring, ax, ay, sc * 1.04, (0.35 + 1.4 * k) * appear, add=True)
        if a['t'] < t < a['t'] + 0.5:
            q = (t - a['t']) / 0.5
            lay = np.zeros((H // 4, W // 4, 3), np.float32)
            cv2.circle(lay, (int(ax / 4), int(ay / 4)), int(P(AV_R) * (1 + 1.3 * q) / 4), a['col'],
                       max(1, int(P(16) / 4)), cv2.LINE_AA)
            f += cv2.resize(cv2.GaussianBlur(lay, (0, 0), P(6) / 4), (W, H)) * 1.6 * (1 - q)
    draw_title(f, ('SHARE IT WITH', 'YOUR FRIENDS'), t, 15.05, P(640), t1=17.82, size=220)
    if t > 17.86:
        f = zoom_blur(f, 0.12 * ei3(seg(t, 17.86, 18.0)), cy=P(2400))
    return f


# ============================================================ 0:18 THEY JOIN
JOIN_T = [18.1, 19.05, 19.6, 20.15]
SEAT_COLS = [RED, GREEN, BLUE, WHITE]


def seats(f, t):
    for i, (tj, col) in enumerate(zip(JOIN_T, SEAT_COLS)):
        x, y = P(540 + i * 360), P(3260)
        k = eo3((t - tj) / 0.15)
        allk = math.exp(-max(0, t - 20.3) / 0.25) * (t > 20.3)
        lay = np.zeros((H // 4, W // 4, 3), np.float32)
        r = int(P(62) / 4)
        cv2.circle(lay, (int(x / 4), int(y / 4)), r, (0.25, 0.25, 0.27), max(1, int(P(8) / 4)), cv2.LINE_AA)
        if k > 0:
            cv2.circle(lay, (int(x / 4), int(y / 4)), r, tuple(c * k for c in col), -1, cv2.LINE_AA)
        core = cv2.GaussianBlur(lay, (0, 0), P(3) / 4)
        gl = cv2.GaussianBlur(lay, (0, 0), P(40) / 4)
        pop = 1 + 2.5 * math.exp(-max(0, t - tj) / 0.12) * (t > tj)
        y0, y1 = int(max(0, y / 4 - r * 6)), int(min(H // 4, y / 4 + r * 6))
        x0, x1 = int(max(0, x / 4 - r * 6)), int(min(W // 4, x / 4 + r * 6))
        roi = np.zeros((H // 4, W // 4, 3), np.float32)
        roi[y0:y1, x0:x1] = core[y0:y1, x0:x1] * 1.4 + gl[y0:y1, x0:x1] * (1.5 * pop + 2 * allk) * k
        f += cv2.resize(roi, (W, H))


def s_join(t):
    if t < 18.9:
        desk = VARIANT == 'v2'
        scr = clip('rec_code', 60).at(3.0 + (t - 18.0))
        lt = t - 18.0
        pose = [0, 0, lerp(2.85, 2.7, lt / 0.9), lerp(14, 6, lt / 0.9), 6, 0]
        bg = arena('env_45.jpg', t, ox=pose[3] * 10)
        f, _, _ = shot(bg, scr, pose, (960, 503), (W / 2, P(2050)), t, desk=desk, rim=(0.2, 0.3, 0.9))
        if t < 18.12:
            f = zoom_blur(f, 0.12 * (1 - seg(t, 18.0, 18.12)), cy=P(2400))
    else:
        mt = remap(t, [(18.9, 2.2), (19.5, 3.4), (20.0, 4.9)]) if t < 20.0 else remap(t, [(20.0, 5.15), (21.0, 5.75)])
        img = clip('mov').at(mt - 1.0)
        fx = 0.47 if t < 20.0 else lerp(0.6, 0.66, seg(t, 20.0, 21.0))
        zw = lerp(2.5, 2.7, seg(t, 18.9, 20.0)) if t < 20.0 else lerp(2.45, 2.6, seg(t, 20.0, 21.0))
        f = band(img, focus=(fx, 0.5), zoom=zw, bg_dim=0.3)
        if t < 19.0:
            f += 0.5 * (1 - seg(t, 18.9, 19.0))
    for tj, col in zip(JOIN_T[1:], SEAT_COLS[1:]):
        if tj <= t < tj + 0.35:
            ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None] ** 2
            f += ys * np.array(col, np.float32) * 0.35 * (1 - (t - tj) / 0.35)
    atmos(f, t, k=0.8)
    seats(f, t)
    draw_title(f, ('THEY JOIN',), t, 18.05, P(640), t1=20.9, size=270)
    return f


# ============================================================ 0:21 EVERYONE'S IN
def s_all_in(t):
    img = clip('env_top').at(t - 21.0 + 0.6)
    lt = t - 21.0
    f = band(img, focus=(0.523, 0.545), zoom=lerp(3.3, 3.65, eio(lt / 2.0)), angle=lerp(0, -7, eio(lt / 2.0)),
             bg_dim=0.35)
    if t < 21.12:
        f += 0.5 * (1 - seg(t, 21.0, 21.12))
    k = ei3(seg(t, 22.45, 23.0))
    if k > 0:  # the board lights up and becomes the cut
        f += np.maximum(f - 0.35, 0) * 3.0 * k + 0.9 * k * k
    atmos(f, t, k=0.7)
    draw_title(f, ("EVERYONE'S IN",), t, 21.05, P(620), t1=22.7, size=260)
    return f


# ============================================================ 0:23 LET THE GAME BEGIN
H_CUTS = [  # start, end, clip, clip_start_abs, abs_at_start, speed, focus, width-zoom, pc yaw
    (23.0, 23.85, 'tra', 4.9, 5.35, 1.5, (0.33, 0.37), 2.3, -9),
    (23.85, 24.6, 'gp1a', 13.6, 14.0, 2.0, (0.55, 0.42), 2.0, 8),
    (24.6, 25.35, 'gp1b', 21.0, 21.6, 2.0, (0.36, 0.45), 2.2, -7),
    (25.35, 26.0, 'env_low', 38.5, 39.75, 1.0, (0.5, 0.55), 2.4, 6),
]


def cut_frame(t, c):
    t0, t1, name, base, a0, sp, foc, zw, yaw = c
    img = clip(name).at(a0 - base + (t - t0) * sp)
    if VARIANT == 'v2' and name != 'env_low':  # gameplay plays on the PC screen
        return gameplay_on_pc(img, t, t0, yaw)
    z = zw * (1 + 0.14 * (1 - eo3((t - t0) / 0.2))) * (1 + 0.05 * (t - t0))
    return band(img, focus=foc, zoom=z, bg_dim=0.32)


def s_begin(t):
    for i, c in enumerate(H_CUTS):
        if c[0] <= t < c[1] or (i == len(H_CUTS) - 1 and t >= c[0]):
            f = cut_frame(t, c)
            if i == 0 and t > 23.68:  # the capture smoke wipes into the next shot
                nxt = cut_frame(H_CUTS[1][0] + 0.001, H_CUTS[1])
                k = eio(seg(t, 23.68, 23.85))
                m = np.clip((f.mean(2) - (0.85 - 0.8 * k)) * 4, 0, 1)
                m = np.clip(cv2.GaussianBlur(m, (0, 0), P(30))[..., None] * (0.5 + 1.5 * k), 0, 1)
                f = f * (1 - m) + nxt * m + m * (1 - m) * 1.2
            break
    if t < 23.25:
        f += 1.0 * (1 - eo3(seg(t, 23.0, 23.25)))
    atmos(f, t, k=0.7)
    draw_title(f, ('LET THE GAME', 'BEGIN'), t, 23.0, P(600), t1=25.25, size=260, shake=18)
    return f


# ============================================================ 0:26 PLAY WITH YOUR FRIENDS / TRY IT NOW
def s_friends(t):
    warm = 0.0
    v2 = VARIANT == 'v2'
    if t < 27.4:
        lt = t - 26.0
        img = clip('fam', 60).at(lt)
        f = band(img, focus=(lerp(0.36, 0.47, eio(lt / 1.4)), 0.5), zoom=lerp(2.0, 2.15, lt / 1.4), bg_dim=0.3)
        warm = 1.0
    elif t < 27.9:
        img = clip('gp1b').at(1.2 + (t - 27.4) * 1.4)
        f = gameplay_on_pc(img, t, 27.4, 8) if v2 else \
            band(img, focus=(0.38, 0.45), zoom=2.2 * (1 + 0.15 * (1 - eo3((t - 27.4) / 0.15))), bg_dim=0.3)
    elif t < 29.3:
        lt = t - 27.9
        img = clip('fam', 60).at(2.0 + lt)
        f = band(img, focus=(lerp(0.5, 0.53, lt / 1.4), lerp(0.42, 0.33, eio(lt / 1.4))),
                 zoom=lerp(2.0, 2.7, eio(lt / 1.4)), bg_dim=0.3)
        warm = 1.0
    else:
        img = clip('tra').at(5.95 - 4.9 + (t - 29.3))
        f = gameplay_on_pc(img, t, 29.3, -8) if v2 else \
            band(img, focus=(0.31, 0.34), zoom=2.4 * (1 + 0.18 * (1 - eo3((t - 29.3) / 0.15))), bg_dim=0.3)
    if warm:
        f = f * 0.78 * np.array([0.82, 0.95, 1.12], np.float32)
    for tc in (26.0, 27.4, 27.9, 29.3):
        if tc <= t < tc + 0.12:
            f += 0.55 * (1 - (t - tc) / 0.12)
    atmos(f, t, k=0.6 if warm else 0.9)
    draw_title(f, ('PLAY WITH', 'YOUR FRIENDS'), t, 26.05, P(600), t1=29.18, size=240)
    draw_title(f, ('TRY IT NOW',), t, 29.3, P(1920) if not v2 else P(700), size=330, shake=20)
    if t > 29.86:
        f *= 1 - eio(seg(t, 29.86, 30.0)) * 0.85
    return f, warm


# ============================================================ 0:30 END CARD
@functools.lru_cache(maxsize=None)
def end_assets():
    s10 = still('shot10_4.png').astype(np.float32) / 255.0

    def keyed(x0, y0, x1, y1, width, thr=0.16, soft=0.22, min_area=60):
        crop = s10[y0:y1, x0:x1]
        a = np.clip((crop.max(2) - thr) / soft, 0, 1)
        n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 0.3).astype(np.uint8), 8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        a *= keep[lab]
        rgb = np.clip(crop / np.maximum(a[..., None], 1e-3), 0, 1) * (a[..., None] > 0)
        img = np.dstack([rgb, a])
        s = P(width) / img.shape[1]
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    logo = keyed(120, 828, 1005, 1045, 1900)
    badges = s10[1240:1326, 160:905]
    s = P(1560) / badges.shape[1]
    badges = cv2.resize(badges, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    badges = np.dstack([badges, np.ones(badges.shape[:2], np.float32)])
    url = keyed(305, 1395, 760, 1458, 980, thr=0.12, soft=0.3, min_area=4)
    rng = np.random.default_rng(5)
    ys, xs = np.nonzero(logo[..., 3] > 0.5)
    n = min(len(xs), max(400, int(11000 * SC * SC)))
    pick = rng.choice(len(xs), n, replace=False)
    tx = xs[pick] - logo.shape[1] / 2 + P(1080)
    ty = ys[pick] - logo.shape[0] / 2 + P(1300)
    ang = rng.uniform(0, 2 * math.pi, n)
    rad = rng.uniform(P(700), P(1700), n)
    return dict(logo=logo, badges=badges, url=url, tx=tx.astype(np.float32), ty=ty.astype(np.float32),
                sx=(P(1080) + np.cos(ang) * rad).astype(np.float32),
                sy=(P(1300) + np.sin(ang) * rad * 1.3).astype(np.float32),
                delay=rng.uniform(0, 0.35, n).astype(np.float32), swirl=rng.uniform(1.0, 2.2, n).astype(np.float32))


LOGO_T = 31.2


def s_end(t):
    f = np.zeros((H, W, 3), np.float32)
    lay = np.zeros((H // 8, W // 8, 3), np.float32)
    cv2.circle(lay, (W // 16, int(P(1300) / 8)), int(W * 0.45 / 8), (0.05, 0.08, 0.16), -1)
    f += cv2.resize(cv2.GaussianBlur(lay, (0, 0), W / 8 * 0.2), (W, H)) * eo3(seg(t, 30.6, 31.4))
    atmos(f, t, k=0.45)
    E = end_assets()
    u = np.clip((t - 30.1 - E['delay']) / 0.85, 0, 1)
    live = u < 1
    if live.any():
        e = u * u * (3 - 2 * u)
        a = (1 - e) * E['swirl']
        dx, dy = E['sx'] - E['tx'], E['sy'] - E['ty']
        x = E['tx'] + (dx * np.cos(a) - dy * np.sin(a)) * (1 - e)
        y = E['ty'] + (dx * np.sin(a) + dy * np.cos(a)) * (1 - e)
        cols = np.repeat(np.array([[0.55, 0.85, 1.0]], np.float32), len(x), 0) * (0.35 + 0.65 * u)[:, None]
        splat(f, x[live], y[live], cols[live], sigma_glow=6, glow=0.7)
    place_post(E['logo'], P(1080), P(1300), 1.0, eo3(seg(t, LOGO_T - 0.12, LOGO_T + 0.15)))
    if t > LOGO_T:
        f += 0.25 * math.exp(-(t - LOGO_T) / 0.12)
    draw_title(f, ('QUESS - 4 PLAYER CHESS',), t, 31.6, P(1680), size=150, shake=8)
    kb = eo3(seg(t, 31.95, 32.3))
    place_post(E['badges'], P(1080), P(2100) + P(60) * (1 - kb), 1.0, kb)
    ku = eo3(seg(t, 32.15, 32.5))
    place_post(E['url'], P(1080), P(2360) + P(50) * (1 - ku), 1.0, ku)
    kd = eo3(seg(t, 32.35, 32.7))
    place_post(text_img('An internet connection is needed to play.', 66, fg=(200, 200, 205)), P(1080), P(3040), 1.0, kd)
    return f


# ============================================================ assembly
SHAKES = [(0.95, 22), (1.72, 12), (1.92, 12), (3.05, 14), (6.0, 14), (7.05, 14), (7.85, 10), (CLICK_T, 22),
          (12.05, 12), (15.05, 12), (BURST_T, 16), (18.05, 14), (18.9, 12), (19.05, 8), (19.6, 8), (20.15, 12),
          (21.05, 12), (23.0, 40), (23.85, 20), (24.6, 20), (25.35, 20), (26.05, 12), (27.4, 24), (27.9, 10),
          (29.3, 46), (LOGO_T, 16), (31.6, 12)]


def shake_at(t):
    dx = dy = 0.0
    for tc, amp in SHAKES:
        if 0 <= t - tc < 0.45:
            k = t - tc
            a = P(amp) * math.exp(-k / 0.09)
            dx += a * math.sin(k * 87 + tc)
            dy += a * 0.7 * math.cos(k * 71 + tc * 3)
    return dx, dy


def render(t_out):
    t = new2old(t_out)
    warm = 0.0
    if t < 3.0:
        f = s_hook(t)
    elif t < 7.0:
        f = s_choose(t)
    elif t < 12.0:
        f = s_create(t)
    elif t < 15.0:
        f = s_code(t)
    elif t < 18.0:
        f = s_share(t)
    elif t < 21.0:
        f = s_join(t)
    elif t < 23.0:
        f = s_all_in(t)
    elif t < 26.0:
        f = s_begin(t)
    elif t < 30.0:
        f, warm = s_friends(t)
    else:
        f = s_end(t)
    if 2.92 <= t < 3.12:  # hook push resolves into a flash
        f += 1.1 * math.exp(-abs(t - 3.0) / 0.05)
    dx, dy = shake_at(t)
    if abs(dx) + abs(dy) > 0.3:
        M = np.array([[1.02, 0, -0.01 * W + dx], [0, 1.02, -0.01 * H + dy]], np.float32)
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return flush_post(grade(f, t_out, warm=warm, leak=0.12 if warm else 0.09))
