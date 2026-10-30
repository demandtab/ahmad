"""Soundtrack for the reel cut: ONE track (Mixkit "Games Music" #706, its lift on 24.0) + four soft SFX.

  python3 mix_reel.py out.wav

Level map (dB vs full): 0-3 full from the first frame | 3-15 the track's own build | 15-24 full |
24-28 climax (drop exactly at 24.0) | 28-30 -6 under the logo | 30-34.5 stays -6, fades to zero by 34.5.
SFX (3 kinds only): piece tick on every move landing, capture hit on "WAIT.", logo chime. No whooshes.
"""
import os
import sys

import numpy as np

os.environ.setdefault('SC', '0.25')
os.environ['SCENES_MOD'] = 'scenes_reel'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'trailer'))
from mix import SR, load, sfx, put  # noqa: E402
import scenes_reel as S  # noqa: E402
from mix_pm import env, finish  # noqa: E402

DUR = S.DURATION
K = S.K
TRACK_LIFT = 41.192   # #706: the breakdown ends and the full section lifts here (onset + RMS 0.17 -> 0.29)
MAP = [(0.0, 0), (23.95, 0), (24.0, 2.5), (K['logo'] - 0.1, 2.5), (K['logo'] + 0.3, -6), (DUR - 1.1, -6), (DUR - 0.2, -90), (DUR, -90)]


P_BEAT = 0.4296


def music():
    """One track, beat-edited so the middle never dips: the track's quieter breakdown (just before its lift) is
    replaced by a full-energy section of the same track; the lift still lands on 24.0. Cuts sit on bar lines."""
    n = int(DUR * SR)
    m = load(os.path.join(S.C.ROOT, 'music', '706.mp3'))
    off = TRACK_LIFT - 24.0                                   # video t -> track t + off (sections A and C)
    ta = 24.0 - 28 * P_BEAT                                   # 11.97 s: splice into the full-energy repeat
    sb = off + ta - 28 * P_BEAT                               # same bar phase, 28 beats earlier in the track
    xf = int(0.04 * SR)
    bus = np.zeros((n, 2), np.float32)
    def seg(v0, v1, s0):
        i0, i1 = int(v0 * SR), min(n, int(v1 * SR) + xf)
        a = m[int(s0 * SR):int(s0 * SR) + (i1 - i0)].copy()
        r = np.ones(len(a), np.float32)
        if v0 > 0:
            r[:xf] = np.linspace(0, 1, xf)
        if v1 < DUR:
            r[-xf:] = np.linspace(1, 0, xf)
        bus[i0:i0 + len(a)] += a * r[:, None]
    seg(0.0, ta, off)
    seg(ta, 24.0, sb)
    seg(24.0, DUR, off + 24.0)
    k = int(0.03 * SR)
    bus[:k] *= np.linspace(0, 1, k)[:, None]
    return bus * env(MAP, n)


def design():
    n = int(DUR * SR)
    bus = np.zeros((n, 2), np.float32)
    for tl in S.LANDS:                                         # piece tick on every landing
        put(bus, sfx('1113'), tl, 0.32, align='peak')
    put(bus, sfx('2150'), K['hit'], 0.40, align='peak')      # capture hit, under the music
    put(bus, sfx('3109'), K['logo'], 0.35, align='peak')     # logo chime
    return bus


if __name__ == '__main__':
    out = sys.argv[1]
    finish(music() + design(), out)
    print('wrote', out)
