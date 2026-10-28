"""Shot-by-shot timeline of the Quess vertical trailer (34.0 s @ 30 fps)."""
import functools
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import (FPS, H, P, SC, W, Embers, arena, band, clip, clamp01, cover, device, device_shot,
                    dir_blur, draw_title, ei3, eio, eo3, font, glow_add, grade, lerp, piecewise, place, remap,
                    screen_glow, seg, splat, still, text_img, title_img, zoom_blur)

DURATION = 34.0

# army colours (BGR)
RED = (0.16, 0.18, 1.0)
GREEN = (0.30, 1.0, 0.25)
BLUE = (1.0, 0.55, 0.12)
WHITE = (1.0, 0.97, 0.92)
GOLD = (0.30, 0.74, 1.0)

DUST = Embers(3, 90, size=(1.5, 3.5), speed=(20, 80), color=(0.6, 0.65, 0.7), gain=0.35)
EMB = Embers(1, 130, size=(2, 5), speed=(90, 300), color=(0.18, 0.5, 1.0), gain=0.9)
BOKEH = Embers(2, 14, size=(20, 48), speed=(260, 620), color=(0.2, 0.55, 1.0), down=6, blur=5, gain=0.3)


def atmos(frame, t, ox=0.0, k=1.0):
    DUST.draw(frame, t, ox * 0.4, gain=k)
    EMB.draw(frame, t, ox, gain=k)
    BOKEH.draw(frame, t, ox * 1.8, gain=k)


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
    # the two new armies light up as the board opens out
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
PANELS = {'cheq': (338, 238, 722, 911), 'quess': (768, 238, 1152, 911), 'baq': (1188, 221, 1592, 931)}


def ignite(t, t0, nxt=None, peak=1.6, hold=0.55):
    if t < t0:
        return 0.0
    k = peak * math.exp(-(t - t0) / 0.25) + hold * eo3((t - t0) / 0.1)
    if nxt is not None and t > nxt:
        k *= lerp(1.0, 0.35, eo3((t - nxt) / 0.2))
    return k


@functools.lru_cache(maxsize=None)
def glow_version(lines, size, fg, sigma):
    img = title_img(lines, size, (0, 0, 0), fg, bar_on=False).copy()
    a = cv2.GaussianBlur(img[..., 3], (0, 0), P(sigma))
    rgb = np.ones_like(img[..., :3]) * (np.array(fg[::-1], np.float32) / 255.0)
    return np.dstack([rgb, a])


def game_label(f, t, t0, t1, text, fg, size, cy, glow=1.0):
    if t < t0 or t > t1 + 0.16:
        return
    tau = t - t0
    s = 1 + 0.6 * (1 - eo3(tau / 0.12))
    a = eo3(tau / 0.07) * (1 - eo3((t - t1) / 0.16) if t > t1 else 1)
    place(f, glow_version((text,), size, fg, 40), W / 2, cy, s * 1.02, a * glow * 1.4, add=True)
    draw_title(f, (text,), t, t0, cy, t1=t1, size=size, fg=fg, bar_on=False, shake=14)


def s_choose(t):
    src = min(t - 2.8, 3.55)
    scr = clip('rec_select').at(src).copy()
    kc = ignite(t, 4.2, 5.2)
    kb = ignite(t, 5.2, 6.0)
    kq = ignite(t, 6.0, None, peak=2.4, hold=1.1)
    scr = screen_glow(scr, PANELS['cheq'], GREEN, kc)
    scr = screen_glow(scr, PANELS['baq'], BLUE, kb)
    scr = screen_glow(scr, PANELS['quess'], RED, kq, width=34, inner=0.12 * min(kq, 1.2))
    lt = t - 3.0
    yaw = lerp(-22, -9, eio(lt / 4.0))
    pose = [0, 0, lerp(2.85, 2.65, eio(lt / 4.0)), yaw, 7, 2.5]
    pose = device().aim(pose, (960, 503), (W / 2, P(2000)))
    bg = arena('env_10.5.jpg', t, ox=yaw * 12)
    rim = tuple(np.array(RED) * (0.35 + 0.6 * min(kq, 1.0)))
    f, Hm = device_shot(bg, scr, pose, glow=0.6, rim_light=rim)
    # dramatic side light from the right
    lay = np.zeros((H // 8, W // 8, 3), np.float32)
    cv2.circle(lay, (W // 8, int(H * 0.45 / 8)), int(W * 0.5 / 8), (0.25, 0.3, 0.9), -1)
    f += cv2.resize(cv2.GaussianBlur(lay, (0, 0), W / 8 * 0.25), (W, H)) * 0.35
    atmos(f, t, ox=yaw * 25)
    draw_title(f, ('CHOOSE YOUR GAME',), t, 3.05, P(720), size=210)
    game_label(f, t, 4.2, 5.15, 'CheQuers', (90, 255, 110), 250, P(3020))
    game_label(f, t, 5.2, 5.95, 'BaQugammon', (80, 170, 255), 250, P(3020))
    game_label(f, t, 6.0, 9.0, 'Quess', (255, 60, 50), 380, P(2990), glow=1.6)
    if t >= 6.1:
        draw_title(f, ('4-PLAYER CHESS',), t, 6.1, P(3340), size=120, shake=10)
    if t > 6.82:  # whip out
        k = ei3(seg(t, 6.82, 7.0))
        f = dir_blur(f, P(900) * k, 0, n=7)
    return f


# ============================================================ 0:07 CREATE A PRIVATE ROOM
ENVELOPE = (320, 446, 665, 711)
SEAL = (455, 548, 525, 618)
BUTTON = (343, 798, 643, 868)
BTN_C = (493, 833)
CLICK_T = 11.67


def s_create(t):
    d = device()
    if t < 7.8:
        scr = clip('rec_mode').at(remap(t, [(7.0, 8.4), (7.8, 9.75)]) - 7.6)
    elif t < 8.45:
        scr = clip('rec_private').at(remap(t, [(7.8, 12.95), (8.45, 14.42)]) - 12.9)
    elif t < CLICK_T:
        abs_t = remap(t, [(8.45, 37.15), (10.3, 39.0), (CLICK_T, 39.675)])
        scr = clip('rec_env', 60).at(abs_t - 36.0)
    else:
        scr = clip('rec_code', 60).at(t - CLICK_T + 0.01)
    scr = scr.copy()
    if 8.45 <= t < CLICK_T:
        pulse = 0.55 + 0.35 * math.sin((t - 8.45) * 7.0)
        scr = screen_glow(scr, ENVELOPE, GOLD, 0.4 * pulse * eo3((t - 8.45) / 0.3), width=26)
        scr = screen_glow(scr, SEAL, (0.2, 0.45, 1.0), 0.45 * pulse, width=30, inner=0.25)
        scr = screen_glow(scr, BUTTON, (1.0, 0.8, 0.5), 0.6 * eio(seg(t, 10.2, CLICK_T)), width=16)
    if t < 8.45:
        lt = t - 7.0
        pose = [0, 0, 2.6, lerp(17, 8, lt / 1.45), 4, lerp(-7, 0, eo3(lt / 0.2))]
        pose = d.aim(pose, (960, 503), (W / 2, P(2000)))
    else:
        u = eio(seg(t, 8.45, 11.55))
        z = lerp(2.6, 1.5, u)
        if t > CLICK_T:
            z = lerp(1.5, 1.85, eo3(seg(t, CLICK_T, 12.0)))
        pose = [0, 0, z, lerp(8, 2, u), lerp(4, 2, u), 0]
        sp = (lerp(960, BTN_C[0], u), lerp(503, BTN_C[1], u))
        pose = d.aim(pose, sp, (W / 2, lerp(P(2000), P(2250), u)))
    bg = arena('env_45.jpg', t, ox=pose[3] * 10)
    f, Hm = device_shot(bg, scr, pose, glow=0.6, rim_light=(0.2, 0.45, 0.8))
    atmos(f, t, ox=pose[3] * 20)
    draw_title(f, ('CREATE A', 'PRIVATE ROOM'), t, 7.05, P(760), t1=11.5, size=230)
    if t < 7.12:  # whip in
        f = dir_blur(f, P(900) * (1 - eo3(seg(t, 7.0, 7.12))), 0, n=7)
    if t >= CLICK_T - 0.02:  # light burst from the tapped button
        bx, by = d.map_screen(Hm, [BTN_C])[0]
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
    # char column ranges
    edges = [pad - b[0] + f.getlength(CODE[:i]) for i in range(len(CODE) + 1)]
    # gold face: vertical gradient
    g = np.linspace(1.0, 0.55, h, dtype=np.float32)[:, None]
    rgb = np.dstack([0.22 * g + 0.08, 0.82 * g + 0.25, np.ones_like(g) * 1.3]) * np.ones((h, w, 1), np.float32)
    img = np.dstack([rgb, mask])
    glow = np.dstack([np.ones((h, w, 3), np.float32) * np.array(GOLD, np.float32),
                      cv2.GaussianBlur(mask, (0, 0), P(18))])
    # particle targets per char
    rng = np.random.default_rng(11)
    ys, xs = np.nonzero(mask > 0.5)
    per = max(60, int(700 * SC * SC))
    pts = []
    for i in range(len(CODE)):
        sel = (xs >= edges[i]) & (xs < edges[i + 1])
        idx = np.nonzero(sel)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(per, len(idx)), replace=False)
        for j in pick:
            pts.append((i, xs[j], ys[j]))
    pts = np.array(pts, np.float32)
    ox, oy = P(HOLO_C[0]) - w / 2, P(HOLO_C[1]) - h / 2
    tx, ty = pts[:, 1] + ox, pts[:, 2] + oy
    starts = rng.uniform(0, 1, (len(pts), 2)).astype(np.float32)
    jit = rng.uniform(0, 1, len(pts)).astype(np.float32)
    return dict(img=img, glow=glow, mask=mask, edges=edges, w=w, h=h, ci=pts[:, 0].astype(int), tx=tx, ty=ty,
                starts=starts, jit=jit, rng_dir=rng.normal(0, 1, (len(pts), 2)).astype(np.float32),
                rnd=rng.uniform(0, 1, len(pts)).astype(np.float32))


def char_arrivals():
    return [CHAR_T0 + i * CHAR_DT + CHAR_FLY for i in range(len(CODE))]


def draw_holo_crisp(f, t, alpha_cols, flicker=1.0, boost=1.0):
    A = holo_assets()
    img = A['img'].copy()
    img[..., 3] *= alpha_cols[None, :] * flicker
    gl = A['glow'].copy()
    gl[..., 3] *= alpha_cols[None, :] * flicker
    place(f, gl, P(HOLO_C[0]), P(HOLO_C[1]), 1.0, 0.9 * boost, add=True)
    place(f, img, P(HOLO_C[0]), P(HOLO_C[1]), 1.0, 1.0)
    if boost > 1:
        place(f, img, P(HOLO_C[0]), P(HOLO_C[1]), 1.0, boost - 1, add=True)


def col_alpha(t):
    A = holo_assets()
    cols = np.zeros(A['w'], np.float32)
    for i, ta in enumerate(char_arrivals()):
        x0, x1 = int(A['edges'][i]), int(math.ceil(A['edges'][i + 1]))
        cols[max(0, x0):x1] = eo3((t - ta + 0.05) / 0.18)
    return cols


def code_pose(t):
    lt = t - 12.0
    pose = [0, 0, 2.7, lerp(-9, 4, eio(lt / 3.0)), 12, 0]
    if t > 15.0:
        pose[2] = 2.7
    return pose


def s_code(t):
    d = device()
    src = remap(t, [(12.0, 0.33), (12.6, 0.93), (15.0, 2.13)])
    scr = clip('rec_code', 60).at(src).copy()
    k_box = eo3(seg(t, 12.55, 12.75))
    scr = screen_glow(scr, CODE_BOX, GOLD, 0.55 * k_box, width=18)
    pose = code_pose(t)
    pose = d.aim(pose, (960, 503), (W / 2, P(2600)))
    bg = arena('env_45.jpg', t, ox=pose[3] * 10, dim=0.27)
    f, Hm = device_shot(bg, scr, pose, glow=0.55, rim_light=(0.25, 0.6, 0.9))
    atmos(f, t, ox=pose[3] * 20, k=0.8)
    if t < 12.12:  # coming out of the click flash
        f += 0.6 * (1 - seg(t, 12.0, 12.12))
    A = holo_assets()
    box = d.map_screen(Hm, [(CODE_BOX[0], CODE_BOX[1]), (CODE_BOX[2], CODE_BOX[1]),
                            (CODE_BOX[2], CODE_BOX[3]), (CODE_BOX[0], CODE_BOX[3])])
    # projection beam from the code box up to the hologram
    kb = eo3(seg(t, 12.6, 12.9))
    if kb > 0:
        lay = np.zeros((H // 4, W // 4, 3), np.float32)
        hw = A['w'] / 2
        poly = np.array([box[0], box[1], (P(HOLO_C[0]) + hw, P(HOLO_C[1]) + A['h'] * 0.3),
                         (P(HOLO_C[0]) - hw, P(HOLO_C[1]) + A['h'] * 0.3)]) / 4
        cv2.fillConvexPoly(lay, poly.astype(np.int32), GOLD, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), P(30) / 4)
        f += cv2.resize(lay, (W, H)) * 0.16 * kb
    # particles lifting off the screen, one character at a time
    ci = A['ci']
    t0 = CHAR_T0 + ci * CHAR_DT + A['jit'] * 0.05
    u = np.clip((t - t0) / CHAR_FLY, 0, 1)
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
    draw_holo_crisp(f, t, col_alpha(t), flicker=0.93 + 0.07 * math.sin(t * 41))
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
    d = device()
    pose = code_pose(t)
    pose = d.aim(pose, (960, 503), (W / 2, P(2600) + ei3(seg(t, 15.0, 15.45)) * P(2400)))
    scr = clip('rec_code', 60).at(2.13 + (t - 15.0)).copy()
    bg = arena('env_45.jpg', t, ox=pose[3] * 10, dim=0.27)
    if t < 15.45:
        f, Hm = device_shot(bg, scr, pose, glow=0.55, rim_light=(0.25, 0.6, 0.9))
        f = dir_blur(f, 0, P(260) * ei3(seg(t, 15.0, 15.45)))
    else:
        f = bg.copy()
        ys = np.arange(H, dtype=np.float32)
        f *= (1 - 0.5 * np.clip((ys - P(2850)) / P(500), 0, 1))[:, None, None]
    atmos(f, t, ox=-20 * (t - 15), k=0.9)
    A = holo_assets()
    if t < BURST_T:
        charge = 1 + 1.2 * eio(seg(t, 15.15, BURST_T))
        draw_holo_crisp(f, t, np.ones(A['w'], np.float32), boost=charge)
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
                cols = np.array(GOLD, np.float32)[None] * (gain * 0.9)
                cols = np.repeat(cols, len(x), 0)
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
        if t > a['t'] and t < a['t'] + 0.5:  # shock ring
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
        d = device()
        scr = clip('rec_code', 60).at(3.0 + (t - 18.0)).copy()
        scr = screen_glow(scr, (380, 278, 712, 330), GOLD, 0.7 * eo3(seg(t, 18.1, 18.3)), width=18)
        lt = t - 18.0
        pose = [0, 0, lerp(2.85, 2.7, lt / 0.9), lerp(14, 6, lt / 0.9), 6, 0]
        pose = d.aim(pose, (960, 503), (W / 2, P(2050)))
        bg = arena('env_45.jpg', t, ox=pose[3] * 10)
        f, _ = device_shot(bg, scr, pose, glow=0.6, rim_light=(0.2, 0.3, 0.9))
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
            k = 1 - (t - tj) / 0.35
            ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None] ** 2
            f += ys * np.array(col, np.float32) * 0.35 * k
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
H_CUTS = [  # start, end, clip, clip_start_abs, abs_at_start, speed, focus, width-zoom
    (23.0, 23.85, 'tra', 4.9, 5.35, 1.5, (0.33, 0.37), 2.3),
    (23.85, 24.6, 'gp1a', 13.6, 14.0, 2.0, (0.55, 0.42), 2.0),
    (24.6, 25.35, 'gp1b', 21.0, 21.6, 2.0, (0.36, 0.45), 2.2),
    (25.35, 26.0, 'env_low', 38.5, 39.75, 1.0, (0.5, 0.55), 2.4),
]


def cut_frame(t, c):
    t0, t1, name, base, a0, sp, foc, zw = c
    img = clip(name).at(a0 - base + (t - t0) * sp)
    z = zw * (1 + 0.14 * (1 - eo3((t - t0) / 0.2))) * (1 + 0.05 * (t - t0))
    return band(img, focus=foc, zoom=z, bg_dim=0.32), img


def s_begin(t):
    for i, c in enumerate(H_CUTS):
        if c[0] <= t < c[1] or (i == len(H_CUTS) - 1 and t >= c[0]):
            f, _ = cut_frame(t, c)
            # the capture smoke of the first shot wipes into the next
            if i == 0 and t > 23.68:
                nxt, _ = cut_frame(H_CUTS[1][0] + 0.001, H_CUTS[1])
                lum = f.mean(2)
                k = eio(seg(t, 23.68, 23.85))
                m = np.clip((lum - (0.85 - 0.8 * k)) * 4, 0, 1)
                m = cv2.GaussianBlur(m, (0, 0), P(30))[..., None]
                m = np.clip(m * (0.5 + 1.5 * k), 0, 1)
                f = f * (1 - m) + nxt * m + m * (1 - m) * 1.2
            break
    if t < 23.25:
        f += 1.0 * (1 - eo3(seg(t, 23.0, 23.25)))
    atmos(f, t, k=0.7)
    draw_title(f, ('LET THE GAME', 'BEGIN'), t, 23.0, P(600), t1=25.25, size=260, shake=26)
    return f


# ============================================================ 0:26 PLAY WITH YOUR FRIENDS / TRY IT NOW
I_CUTS = [
    (26.0, 27.4, 'fam', 'fam'),
    (27.4, 27.9, 'gp1b', 'hit'),
    (27.9, 29.3, 'fam', 'fam2'),
    (29.3, 30.0, 'tra', 'hit2'),
]


def s_friends(t):
    warm = 0.0
    if t < 27.4:
        lt = t - 26.0
        img = clip('fam', 60).at(lt)
        f = band(img, focus=(lerp(0.36, 0.47, eio(lt / 1.4)), 0.5), zoom=lerp(2.0, 2.15, lt / 1.4), bg_dim=0.3)
        warm = 1.0
    elif t < 27.9:
        img = clip('gp1b').at(1.2 + (t - 27.4) * 1.4)
        f = band(img, focus=(0.38, 0.45), zoom=2.2 * (1 + 0.15 * (1 - eo3((t - 27.4) / 0.15))), bg_dim=0.3)
    elif t < 29.3:
        lt = t - 27.9
        img = clip('fam', 60).at(2.0 + lt)
        f = band(img, focus=(lerp(0.5, 0.53, lt / 1.4), lerp(0.42, 0.33, eio(lt / 1.4))),
                 zoom=lerp(2.0, 2.7, eio(lt / 1.4)), bg_dim=0.3)
        warm = 1.0
    else:
        img = clip('tra').at(5.95 - 4.9 + (t - 29.3))
        f = band(img, focus=(0.31, 0.34), zoom=2.4 * (1 + 0.18 * (1 - eo3((t - 29.3) / 0.15))), bg_dim=0.3)
    if warm:
        f = f * 0.78
        f *= np.array([0.82, 0.95, 1.12], np.float32)
    for tc in (26.0, 27.4, 27.9, 29.3):
        if tc <= t < tc + 0.12:
            f += 0.55 * (1 - (t - tc) / 0.12)
    atmos(f, t, k=0.6 if warm else 0.9)
    draw_title(f, ('PLAY WITH', 'YOUR FRIENDS'), t, 26.05, P(600), t1=29.18, size=240)
    draw_title(f, ('TRY IT NOW',), t, 29.3, P(1920), size=330, shake=34)
    if t > 29.86:
        f *= 1 - eio(seg(t, 29.86, 30.0)) * 0.85
    return f, warm


# ============================================================ 0:30 END CARD
@functools.lru_cache(maxsize=None)
def end_assets():
    s10 = still('shot10_4.png').astype(np.float32) / 255.0

    def keyed(x0, y0, x1, y1, width, thr=0.16, soft=0.22, min_area=60):
        crop = s10[y0:y1, x0:x1]
        lum = crop.max(2)
        a = np.clip((lum - thr) / soft, 0, 1)
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
    bmask = np.ones(badges.shape[:2], np.float32)
    badges = np.dstack([badges, bmask])
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
                sx=(P(1080) + np.cos(ang) * rad).astype(np.float32), sy=(P(1300) + np.sin(ang) * rad * 1.3).astype(np.float32),
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
    k = eo3(seg(t, LOGO_T - 0.12, LOGO_T + 0.15))
    place(f, E['logo'], P(1080), P(1300), 1.0, k)
    if t > LOGO_T:
        fl = math.exp(-(t - LOGO_T) / 0.12)
        place(f, E['logo'], P(1080), P(1300), 1.0, 1.4 * fl, add=True)
        f += 0.35 * fl
        q = seg(t, 31.4, 32.1)  # light sweep across the logo
        if 0 < q < 1:
            logo = E['logo']
            hh, ww = logo.shape[:2]
            xs = np.arange(ww, dtype=np.float32)
            band_ = np.exp(-((xs - q * ww * 1.3 + ww * 0.15) / (ww * 0.05)) ** 2)
            sweep = logo.copy()
            sweep[..., 3] *= band_[None, :]
            place(f, sweep, P(1080), P(1300), 1.0, 1.2, add=True)
    draw_title(f, ('QUESS - 4 PLAYER CHESS',), t, 31.6, P(1700), size=150, shake=12)
    kb = eo3(seg(t, 31.95, 32.3))
    place(f, E['badges'], P(1080), P(2070) + P(60) * (1 - kb), 1.0, kb)
    ku = eo3(seg(t, 32.15, 32.5))
    place(f, E['url'], P(1080), P(2330) + P(50) * (1 - ku), 1.0, ku)
    kd = eo3(seg(t, 32.35, 32.7))
    place(f, text_img('An internet connection is needed to play.', 66, fg=(200, 200, 205)),
          P(1080), P(3040), 1.0, kd)
    if t < 30.15:
        f += 0.0
    return f


# ============================================================ assembly
SHAKES = [(0.95, 22), (1.72, 12), (1.92, 12), (3.05, 14), (6.0, 14), (7.05, 14), (CLICK_T, 22), (12.05, 12),
          (15.05, 12), (BURST_T, 16), (18.05, 14), (18.9, 12), (19.05, 8), (19.6, 8), (20.15, 12), (21.05, 12),
          (23.0, 40), (23.85, 20), (24.6, 20), (25.35, 20), (26.05, 12), (27.4, 24), (27.9, 10), (29.3, 46),
          (LOGO_T, 16), (31.6, 12)]


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
    return grade(f, t, warm=warm, leak=0.12 if warm else 0.09, exposure=1.0)
