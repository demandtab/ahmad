"""Soundtrack for the reel cut: ONE track (Mixkit "Games Music" #706, its lift on 24.0) + four soft SFX.

  python3 mix_reel.py out.wav

Level map (dB vs full): 0-3 full from the first frame | 3-15 the track's own build | 15-24 full |
24-28 climax (drop exactly at 24.0) | 28-30 -6 under the logo | 30-34.5 stays -6, fades to zero by 34.5.
SFX (exactly 3): piece tick on the FIRE RED landing, capture hit on "WAIT.", logo chime. No whooshes.
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


def music():
    n = int(DUR * SR)
    bus = np.zeros((n, 2), np.float32)
    put(bus, load(os.path.join(S.C.ROOT, 'music', '706.mp3')), 0.0, 1.0, src_from=TRACK_LIFT - 24.0)
    # steady middle: ride the track's breakdown (about 12-24 s) up to the level of the opening, so the tension holds
    m = bus.mean(1)
    hop = int(0.05 * SR)
    rms = np.array([np.sqrt((m[i:i + hop] ** 2).mean() + 1e-12) for i in range(0, n, hop)])
    rms = np.convolve(rms, np.ones(30) / 30, mode='same')
    ref = float(np.sqrt((m[int(1 * SR):int(9 * SR)] ** 2).mean()))
    tt = np.arange(len(rms)) * 0.05
    g = np.where((tt > 8.5) & (tt < 23.9), np.clip(ref / rms, 1.0, 10 ** (6 / 20)), 1.0)
    g = np.convolve(g, np.ones(10) / 10, mode='same')
    bus *= np.interp(np.arange(n), np.arange(len(g)) * hop, g)[:, None].astype(np.float32)
    k = int(0.03 * SR)
    bus[:k] *= np.linspace(0, 1, k)[:, None]
    return bus * env(MAP, n)


def design():
    n = int(DUR * SR)
    bus = np.zeros((n, 2), np.float32)
    put(bus, sfx('1113'), S.LAND1_T, 0.35, align='peak')     # piece tick: FIRE RED lands
    put(bus, sfx('2150'), K['hit'], 0.40, align='peak')      # capture hit, under the music
    put(bus, sfx('3109'), K['logo'], 0.35, align='peak')     # logo chime
    return bus


if __name__ == '__main__':
    out = sys.argv[1]
    finish(music() + design(), out)
    print('wrote', out)
