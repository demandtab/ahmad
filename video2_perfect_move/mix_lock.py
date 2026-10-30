"""Soundtrack for the LOCKED recipe: ONE track built from Mixkit "Epic Games" (#76), four SFX at locked times.

  python3 mix_lock.py out.wav

Music edit (all cuts on the track's own beat grid, 105.5 BPM, 40 ms crossfades), so the locked map is met:
  00:00-02.95  a full-energy bar from after the drop (drums in from frame one, no quiet drone)
  02.95-35.00  the track's build -> its break -> the DROP exactly at 00:24.00 -> full section
Level map: full | build | full | climax +2.5 dB from 24.0 | -6 dB under the logo from 28.0 | stays dipped,
fades to zero by 34.5. SFX: 3.00 click, 15.30 pop, 20.75 capture hit (under the music), 28.00 chime.
"""
import os
import sys

import numpy as np

os.environ.setdefault('SC', '0.25')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'trailer'))
from mix import SR, load, sfx, put  # noqa: E402
from mix_pm import env, finish  # noqa: E402

DUR = 35.0
DROP_SRC, P_BEAT = 34.307, 0.5689
T1 = 24.0 - 37 * P_BEAT                  # 2.95: splice point, a beat of the grid pinned on 24.0
SEG2_SRC = DROP_SRC - 37 * P_BEAT        # the build, landing on the drop at 24.0
SEG1_SRC = DROP_SRC + 10 * P_BEAT - T1   # full section after the drop, ending on a beat at the splice
MAP = [(0.0, 0), (23.95, 0), (24.0, 2.5), (27.9, 2.5), (28.2, -6), (33.6, -6), (34.5, -90), (35.0, -90)]


def music():
    n = int(DUR * SR)
    m = load(os.path.join(HERE, '..', 'music', '76.mp3'))
    a = m[int(SEG1_SRC * SR):int(SEG1_SRC * SR) + int(T1 * SR) + int(0.04 * SR)].copy()
    b = m[int(SEG2_SRC * SR):int(SEG2_SRC * SR) + n].copy()
    bus = np.zeros((n, 2), np.float32)
    i1 = int(T1 * SR)
    xf = int(0.04 * SR)
    bus[:i1 + xf] += a[:i1 + xf] * np.r_[np.ones(i1), np.linspace(1, 0, xf)][:, None]
    seg = b[:n - i1]
    seg[:xf] *= np.linspace(0, 1, xf)[:, None]
    bus[i1:i1 + len(seg)] += seg
    k = int(0.02 * SR)
    bus[:k] *= np.linspace(0, 1, k)[:, None]
    return bus * env(MAP, n)


def design():
    bus = np.zeros((int(DUR * SR), 2), np.float32)
    put(bus, sfx('900'), 3.00, 0.30, align='peak')      # soft UI click: Private Room appears
    put(bus, sfx('2356'), 15.30, 0.30, align='peak')    # soft pop: first player card only
    put(bus, sfx('2150'), 20.75, 0.35, align='peak')    # capture hit, under the music
    put(bus, sfx('3109'), 28.00, 0.32, align='peak')    # soft logo chime
    return bus


if __name__ == '__main__':
    finish(music() + design(), sys.argv[1])
    print('wrote', sys.argv[1])
