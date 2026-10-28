"""Final videos, pass 2: no glass type, distinct ending animations, frame-exact beat cuts, smooth PC cuts.

  FINAL=mobile | desktop   (render at SC=0.3333 for 720x1280)
"""
import math
import os

import numpy as np

import scenes5 as base5   # sets VARIANT / R3MODE and picks the mobile or desktop timeline
import scenes3 as s3
from common import FPS, H, P, W, band, clip, dir_blur, eio, eo3, lerp, remap, seg

FINAL = base5.FINAL
DURATION = 34.0
_inner = base5.render

# ------------------------------------------------------------------ ending clips: each with its own animation
END_CUTS = [   # start, end, clip, clip time at start, speed, focus, width-zoom, animation
    (23.0, 24.15, 'tra', 0.45, 1.5, (0.33, 0.37), 2.3, 'push'),
    (24.15, 24.72, 'gp1a', 0.4, 2.0, (0.55, 0.42), 2.0, 'slide'),
    (24.72, 25.29, 'gp1b', 0.6, 2.0, (0.36, 0.45), 2.2, 'roll'),
    (25.29, 25.86, 'env_low', 1.25, 1.0, (0.5, 0.55), 2.4, 'pull'),
]


def anim_frame(t, t0, t1, name, a0, sp, foc, zw, kind):
    lt, dur = t - t0, t1 - t0
    img = clip(name).at(a0 + lt * sp)
    ang, dx, dy, blur = 0.0, 0.0, 0.0, 0.0
    if kind == 'push':      # punch in, keep creeping forward
        z = zw * (1 + 0.14 * (1 - eo3(lt / 0.2))) * (1 + 0.05 * lt)
    elif kind == 'slide':   # whip-pan in from the left, settle
        e = eo3(lt / 0.3)
        z, dx = zw * 1.06, -W * 0.42 * (1 - e)
        blur = P(520) * (1 - e) ** 2
    elif kind == 'roll':    # dutch-angle roll across the board
        z, ang = zw * (1.12 - 0.08 * eio(lt / dur)), lerp(-9, 4, eio(lt / dur))
    elif kind == 'pull':    # pull back to reveal
        z = zw * lerp(1.45, 1.0, eo3(lt / dur))
    elif kind == 'rise':    # tilt up from the board to the army towering over it
        z, dy = zw * (1.02 + 0.04 * lt), H * 0.16 * (1 - eio(lt / dur)) - H * 0.04
    elif kind == 'hero':    # slow-motion hero push on the fire army
        z = zw * (1.0 + 0.16 * eo3(lt / dur))
    f = band(img, focus=foc, zoom=z, angle=ang, dx=dx, dy=dy, bg_dim=0.32)
    if blur > 1:
        f = dir_blur(f, blur, 0, n=7)
    return f


def end_begin(t):
    for i, c in enumerate(END_CUTS):
        if c[0] <= t < c[1] or (i == len(END_CUTS) - 1 and t >= c[0]):
            f = anim_frame(t, *c)
            if i == 0 and t > 23.95:   # capture smoke wipes into the next shot
                import cv2
                nxt = anim_frame(END_CUTS[1][0], *END_CUTS[1])
                k = eio(seg(t, 23.95, 24.15))
                m = np.clip((f.mean(2) - (0.85 - 0.8 * k)) * 4, 0, 1)
                m = np.clip(cv2.GaussianBlur(m, (0, 0), P(30))[..., None] * (0.5 + 1.5 * k), 0, 1)
                f = f * (1 - m) + nxt * m + m * (1 - m) * 1.2
            break
    if t < 23.25:
        f += 1.0 * (1 - eo3(seg(t, 23.0, 23.25)))
    for tc in (24.15, 24.72, 25.29):
        if tc <= t < tc + 0.1:
            f += 0.4 * (1 - (t - tc) / 0.1)
    import scenes as old
    old.atmos(f, t, k=0.7)
    return f


_friends = s3._M['friends']


def end_friends(t):
    if 27.0 <= t < 27.57:     # EARTH GREEN army towers over the board (fresh clip, rise)
        f = anim_frame(t, 27.0, 27.57, 'env_earth', 0.3, 1.0, (0.5, 0.42), 1.9, 'rise')
        if t < 27.1:
            f += 0.5 * (1 - (t - 27.0) / 0.1)
        return f, 0.0
    if t >= 28.72:            # TRY IT NOW: FIRE RED tower, slow-motion hero push
        lt = t - 28.72
        f = anim_frame(t, 28.72, 29.86, 'env_fire', 0.2, 0.55, (0.47, 0.45), 1.9, 'hero')
        if lt < 0.12:
            f += 0.55 * (1 - lt / 0.12)
        if t > 29.66:
            f *= 1 - eio(seg(t, 29.66, 29.86)) * 0.85
        return f, 0.0
    return _friends(t)


s3.k_begin = end_begin
s3.k_friends = end_friends
if FINAL == 'mobile':
    import scenes4
    scenes4.k_begin = end_begin

# ------------------------------------------------------------------ frame-exact beat snapping
HB = 60.0 / 105 / 2          # half-beat of Epic Games (105 BPM)
NOMINAL = [3.58, 4.72, 5.86, 7.0, 8.15, 8.72, 9.86, 11.0, 12.15, 12.72, 13.29, 14.43, 15.0, 15.57, 15.86, 16.15,
           16.43, 16.72, 17.0, 17.29, 17.86, 18.44, 19.01, 19.58, 20.15, 20.72, 23.0, 24.15, 24.72, 25.29, 25.86,
           27.0, 27.57, 28.72, 29.86, 31.0, 31.57, 32.43, 32.72]


def beat_frame(n):
    true = 23.0 + round((n - 23.0) / HB) * HB
    return round(true * FPS) / FPS


SNAP = [(0.0, 0.0)] + [(beat_frame(n) - 0.5 / FPS, n) for n in NOMINAL] + [(DURATION, DURATION)]


def out2nom(t):
    return remap(t, SNAP)


def nom2out(n):   # for sound: the frame on which nominal event n appears
    return remap(n, [(b, a) for a, b in SNAP]) + 0.5 / FPS


# ------------------------------------------------------------------ smooth cuts
CUTS = {'desktop': [4.72, 5.86, 8.15, 8.72, 12.72, 15.0, 15.86, 17.29, 20.72, 23.0, 24.15, 24.72, 25.29, 25.86,
                    27.0, 27.57, 28.72, 29.86],
        'mobile': [5.86, 8.15, 12.72, 15.0, 17.29, 20.72, 24.15, 24.72, 25.29, 25.86, 27.0, 27.57, 28.72]}[FINAL]


def render(t_out):
    t = out2nom(t_out)
    f = _inner(t)
    for i, c in enumerate(CUTS):   # 2-frame motion-blur bridge on both sides of every cut
        d = abs(t_out - beat_frame(c) + 0.5 / FPS)
        if d < 2.0 / FPS:
            k = 1 - d / (2.0 / FPS)
            amp = P(240) * k
            f = dir_blur(f, amp if i % 2 == 0 else 0, 0 if i % 2 == 0 else amp, n=5)
    return f


if __name__ == '__main__':
    for n in NOMINAL:
        print(f'{n:6.2f} -> beat frame {beat_frame(n):6.3f}  off {abs(n - beat_frame(n)) * 1000:4.0f} ms')
