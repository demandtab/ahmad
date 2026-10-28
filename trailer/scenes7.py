"""Final mobile video: the approved mobile cut with a reworked ending (23.0-29.86).

Ending: four full-frame 9:16 shots of the game's 4K cinematic footage (no letterbox band), joined by
beat-centred dissolves with a zoom-through instead of hard cuts. No couch / earth-giant footage.
"""
import os

os.environ['FINAL'] = 'mobile'
import numpy as np  # noqa: E402

import scenes6 as s6  # noqa: E402
import scenes3 as s3  # noqa: E402
import scenes4  # noqa: E402
import scenes as old  # noqa: E402
from common import P, W, H, clip, cover, eio, eo3, lerp, seg, zoom_blur  # noqa: E402

DURATION = 34.0

# shot: start, clip, clip time at start, speed, focus, zoom (x cover), drift kind
SHOTS = [
    (23.0, 'env_a', 0.05, 0.85, (0.56, 0.52), 1.05, 'push'),     # LET THE GAME BEGIN: all four armies, smoke
    (24.72, 'env_red', 0.05, 0.85, (0.74, 0.52), 1.0, 'pan'),     # all four armies, FIRE RED in front, lateral drift
    (26.43, 'env_tilt', 0.0, 0.65, (0.42, 0.5), 1.05, 'turn'),   # board tilt: FIRE RED vs EARTH GREEN
    (28.72, 'env_fire', 0.2, 0.55, (0.47, 0.44), 1.0, 'hero'),   # TRY IT NOW: fire tower, slow-mo hero push
]
END = 29.86
XF = 0.1   # dissolve half-length (6 frames total), centred on the beat


def shot_frame(i, t):
    t0, name, a0, sp, foc, z, kind = SHOTS[i]
    t1 = SHOTS[i + 1][0] if i + 1 < len(SHOTS) else END
    lt, dur = t - t0, t1 - t0
    img = clip(name).at(a0 + max(lt, -XF) * sp)
    ang, dx = 0.0, 0.0
    if kind == 'push':
        zz = z * (1 + 0.10 * eo3((lt + XF) / (dur + XF)))
    elif kind == 'pan':
        zz, dx = z * 1.06, lerp(W * 0.06, -W * 0.06, eio((lt + XF) / (dur + 2 * XF)))
    elif kind == 'turn':
        zz, ang = z * (1.1 - 0.06 * eio((lt + XF) / (dur + 2 * XF))), lerp(-5, 3, eio((lt + XF) / (dur + 2 * XF)))
    else:
        zz = z * (1 + 0.14 * eo3((lt + XF) / (dur + XF)))
    return cover(img, focus=foc, zoom=zz, angle=ang, dx=dx)


def ending(t):
    i = max(k for k, s in enumerate(SHOTS) if t >= s[0] - XF) if t >= SHOTS[0][0] else 0
    f = shot_frame(i, t)
    if i > 0 and t < SHOTS[i][0] + XF:   # beat-centred dissolve with zoom-through
        b = SHOTS[i][0]
        w = eio((t - (b - XF)) / (2 * XF))
        prev = shot_frame(i - 1, t)
        k = 1 - abs(t - b) / XF
        f = zoom_blur(prev, 0.06 * k) * (1 - w) + zoom_blur(f, 0.06 * k) * w + 0.12 * k
    if t < 23.25:
        f = f + 1.0 * (1 - eo3(seg(t, 23.0, 23.25)))
    if t > END - 0.2:
        f = f * (1 - eio(seg(t, END - 0.2, END)) * 0.85)
    old.atmos(f, t, k=0.6)
    return f


def ending_friends(t):
    return ending(t), 0.0


scenes4.k_begin = ending
s3.k_friends = ending_friends

# captions follow the longer shots
scenes4.CAPTIONS = [c for c in scenes4.CAPTIONS if c[0] not in (('LET THE GAME', 'BEGIN'), ('PLAY WITH', 'YOUR FRIENDS'))]
scenes4.CAPTIONS += [(('LET THE GAME', 'BEGIN'), 23.0, 26.28, 620, 210, False),
                     (('PLAY WITH', 'YOUR FRIENDS'), 26.47, 28.6, 620, 190, False)]
s3.SHAKES = [s for s in s3.SHAKES if s[0] not in (24.15, 24.72, 25.29, 25.9, 27.0, 9.86)] + [(26.47, 8), (9.29, 8)]

# frame-exact beat snapping includes the new 26.43 cut; dissolves replace the blur bridges at the end
s6.NOMINAL = sorted(set(s6.NOMINAL + [26.43, 9.29]) - {24.15, 25.29, 25.86, 27.0, 27.57, 9.86})
s6.SNAP = [(0.0, 0.0)] + [(s6.beat_frame(n) - 0.5 / 30, n) for n in s6.NOMINAL] + [(DURATION, DURATION)]
s6.CUTS = [5.86, 8.15, 12.72, 15.0, 17.29, 20.72]
render = s6.render

# ------------------------------------------------------------------ hook: third army lights up (three friends)
import cv2  # noqa: E402
import math  # noqa: E402

_hook = s3.k_hook
BLUE_T = 2.699   # next half-beat after the white (2.15) and green (2.43) flashes; lands on frame 81


def hook_three(t):
    f = _hook(t)
    if t >= BLUE_T:
        k = math.exp(-(t - BLUE_T) / 0.22)
        if k > 0.01:   # WATER BLUE army at the far side of the board lights up
            lay = np.zeros((H // 8, W // 8, 3), np.float32)
            cv2.ellipse(lay, (int(W * 0.5 / 8), int(H * 0.43 / 8)), (int(W * 0.3 / 8), int(H * 0.08 / 8)), 0, 0, 360,
                        old.BLUE, -1)
            lay = cv2.GaussianBlur(lay, (0, 0), W / 8 * 0.08)
            f = f + cv2.resize(lay, (W, H)) * 0.9 * k
    return f


s3.k_hook = hook_three
s3.SHAKES = s3.SHAKES + [(2.7, 10)]
