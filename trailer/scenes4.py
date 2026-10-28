"""Final cut: the V1 (mobile) / V2 (desktop) look and shots, the revision-3 THEY JOIN lobby,
the original script order, no Hotseat, uploaded logos + outro, Mixkit "Epic Games" beat grid.

  VARIANT=v1  phone mockups        VARIANT=v2  the gaming-PC desk setup for every UI / gameplay shot
"""
import math

import cv2
import numpy as np

import common
import scenes as old
import scenes3 as s3
from common import H, P, W, clip, dir_blur, draw_title, eio, eo3, flush_post, grade, lerp, place_post, remap, seg

old.draw_title = lambda *a, **k: None
DESK = old.VARIANT == 'v2'
DURATION = 34.0

CAPTIONS = [
    (('GOT THREE', 'FRIENDS?'), 1.29, 3.42, 880, 290, True),
    (('CREATE A', 'PRIVATE ROOM'), 3.62, 8.0, 660, 240, False),
    (('CHOOSE YOUR GAME',), 8.2, 12.58, 640, 200, False),
    (('GET YOUR', 'ROOM CODE'), 12.76, 14.86, 640, 240, False),
    (('SHARE IT WITH', 'YOUR FRIENDS'), 15.04, 17.15, 640, 230, False),
    (('THEY JOIN',), 17.33, 20.58, 600, 280, False),
    (("EVERYONE'S IN",), 20.76, 22.86, 640, 290, True),
    (('LET THE GAME', 'BEGIN'), 23.0, 25.72, 620, 270, False),
    (('PLAY WITH', 'YOUR FRIENDS'), 25.9, 28.6, 620, 250, False),
    (('TRY IT NOW',), 28.72, 29.78, 1920 if not DESK else 700, 340, False),
]


def captions(f, t):
    for lines, t0, t1, cy, size, glass in CAPTIONS:
        draw_title(f, lines, t, t0, P(cy), t1=t1, size=s3.fit_size(lines, size), glass=glass, shake=14)


def device(bg, scr, pose, aim_src, aim_tgt, t, light=old.RGB_ROOM, rim=(0.2, 0.45, 0.8), glow=0.6):
    return old.shot(bg, scr, pose, aim_src, aim_tgt, t, desk=DESK, light=light, rim=rim, glow=glow)


# ------------------------------------------------------------------ CREATE A PRIVATE ROOM (no Hotseat on screen)
def k_create(t):
    if t < 5.86:   # the real Private Room screen, framed tight; camera orbits and pushes in as the cursor goes to Create Room
        lt = t - 3.58
        scr = clip('rec_priv2', 60).at(remap(t, [(3.58, 0.1), (5.86, 2.5)]))
        pose = [0, 0, lerp(1.6, 1.25, eio(lt / 2.28)), lerp(14, -6, eio(lt / 2.28)), 5, lerp(-6, 0, eo3(lt / 0.3))]
        bg = old.arena('env_45.jpg', t, ox=pose[3] * 10)
        f, _, _ = device(bg, scr, pose, s3.PRIV_FOCUS, (W / 2, P(2150)), t)
        old.atmos(f, t, ox=pose[3] * 20, k=0.6 if DESK else 1.0)
        if t < 3.7:
            f = dir_blur(f, P(800) * (1 - eo3(seg(t, 3.58, 3.7))), 0, n=7)
        return f
    # envelope screen: push-in and slow-motion tap on Create Room (on the 8.15 beat)
    return old.s_create(remap(t, [(5.86, 8.45), (7.0, 10.3), (8.15, 11.67)]))


# ------------------------------------------------------------------ CHOOSE YOUR GAME (uploaded logos)
LOGO_COL = {'Quess': old.RED, 'CheQuers': old.GREEN, 'BaQugammon': old.BLUE}


def light_now(t):
    col = np.array((0.25, 0.25, 0.6), np.float32)
    for name, t0 in s3.LOGO_T:
        if t >= t0:
            col = np.array(LOGO_COL[name], np.float32) * (1 + 0.8 * math.exp(-(t - t0) / 0.25))
    return col


def k_choose(t):
    light = light_now(t)
    if t < 9.29:   # the V1/V2 orbit around the device with the real game-selection screen
        lt = t - 8.15
        scr = clip('rec_select').at(0.2 + lt * 0.66)
        yaw = lerp(-22, -12, eio(lt / 1.14))
        pose = [0, 0, lerp(2.85, 2.6, eio(lt / 1.14)), yaw, 7, 2.5]
        aim, tgt = (960, 503), (W / 2, P(2000))
    else:          # push in panel to panel while each uploaded logo lands
        src = remap(t, [(9.29, 2.0), (10.999, 2.95), (11.0, 3.1), (12.149, 3.55), (12.15, 1.0), (12.72, 1.57)])   # screen highlights the same game as the logo
        scr = clip('rec_select').at(src)
        pts = [s3.LOGOS[n][2] for n, _ in s3.LOGO_T]
        k1, k2 = eio(seg(t, 10.9, 11.1)), eio(seg(t, 12.05, 12.25))
        fx = lerp(lerp(pts[0][0], pts[1][0], k1), pts[2][0], k2)
        yaw = lerp(-10, 8, eio(seg(t, 9.29, 12.72)))
        pose = [0, 0, 1.75 if not DESK else 1.55, yaw, 6, 0]
        aim, tgt = (fx, 520), (W / 2, P(1850))
    bg = old.arena('env_10.5.jpg', t, ox=yaw * 12)
    f, _, _ = device(bg, scr, pose, aim, tgt, t, light=light * 0.8, rim=tuple(light * 0.5))
    if not DESK:   # coloured side light from the army of the game being picked
        lay = np.zeros((H // 8, W // 8, 3), np.float32)
        cv2.circle(lay, (W // 8, int(H * 0.45 / 8)), int(W * 0.5 / 8), tuple(float(v) for v in light * 0.6), -1)
        f += cv2.resize(cv2.GaussianBlur(lay, (0, 0), W / 8 * 0.25), (W, H)) * 0.35
    if t < 8.35:
        f += 1.0 * (1 - eo3(seg(t, 8.15, 8.35)))
    whip = max(math.exp(-abs(t - 9.29) / 0.05), math.exp(-abs(t - 11.0) / 0.05), math.exp(-abs(t - 12.15) / 0.05))
    if whip > 0.05:
        f = dir_blur(f, P(500) * whip, 0, n=7)
    old.atmos(f, t, ox=yaw * 25, k=0.6 if DESK else 1.0)
    if t >= 9.29:
        ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        f *= 1 - 0.55 * np.clip((ys - 0.66) / 0.18, 0, 1)
        for i, (name, t0) in enumerate(s3.LOGO_T):
            t1 = s3.LOGO_T[i + 1][1] - 0.04 if i + 1 < len(s3.LOGO_T) else 12.66
            if not (t0 <= t < t1 + 0.12):
                continue
            tau = t - t0
            e = 1 - math.exp(-tau / 0.07)
            a = min(1.0, tau / 0.1) * (1 - eio((t - t1) / 0.12) if t > t1 else 1)
            img = s3.logo_img(name)
            if tau < 0.18:
                img = cv2.blur(img, (1, max(1, int(P(30) * (1 - e))) * 2 + 1))
            place_post(img, P(1080), P(3080), 1.08 - 0.08 * e, a)
            if name == 'Quess' and tau > 0.15:
                place_post(s3.label('4-PLAYER CHESS', 70), P(1080), P(3520), 1.0,
                           a * eo3((tau - 0.15) / 0.2))
    return f


# ------------------------------------------------------------------ ROOM CODE / SHARE (V1/V2 shots)
def k_code(t):
    return old.s_code(remap(t, [(12.72, 12.0), (13.29, 13.14), (14.43, 14.34), (15.0, 15.0)]))


def k_share(t):
    return old.s_share(remap(t, [(15.0, 15.0), (15.57, 15.38), (16.15, 16.02), (16.43, 16.32), (16.72, 16.62),
                                 (17.29, 17.0)]))


# ------------------------------------------------------------------ the rest
def k_begin(t):
    return old.s_begin(remap(t, [(23.0, 23.0), (24.15, 23.85), (24.72, 24.6), (25.29, 25.35), (25.86, 26.0)]))


def k_end(t):
    f = common.cover(clip('outro').at(t - 29.86), zoom=1.0)
    place_post(s3.label('QUESS - 4 PLAYER CHESS', 110, 'Anton.ttf', 0.06, (0.97, 0.96, 0.94)),
               P(1080), P(2970), 1.0, eo3(seg(t, 32.43, 32.75)))
    place_post(s3.label('AN INTERNET CONNECTION IS NEEDED TO PLAY.', 42, 'Montserrat_wght_500.ttf', 0.16,
                        (0.8, 0.8, 0.82)), P(1080), P(3190), 1.0, eo3(seg(t, 32.72, 33.05)))
    return f


def render(t):
    warm = 0.0
    if t < 3.58:
        f = s3.k_hook(t)
    elif t < 8.15:
        f = k_create(t)
    elif t < 12.72:
        f = k_choose(t)
    elif t < 15.0:
        f = k_code(t)
    elif t < 17.29:
        f = k_share(t)
    elif t < 20.72:
        f = s3.k_join(t)          # the revision-3 lobby
    elif t < 23.0:
        f = s3.k_all_in(t)
    elif t < 25.86:
        f = k_begin(t)
    elif t < 29.86:
        f, warm = s3.k_friends(t)
    else:
        f = k_end(t)
    if 3.48 <= t < 3.7:
        f += 1.1 * math.exp(-abs(t - 3.58) / 0.05)
    dx, dy = s3.shake_at(t)
    if abs(dx) + abs(dy) > 0.3:
        M = np.array([[1.02, 0, -0.01 * W + dx], [0, 1.02, -0.01 * H + dy]], np.float32)
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    captions(f, t)
    if t >= 29.86:
        f = np.clip(f, 0, 1)
    else:
        f = grade(f, t, warm=warm, leak=0.12 if warm else 0.09)
    return flush_post(f)
