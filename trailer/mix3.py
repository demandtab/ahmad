"""Soundtrack for revision 3: Mixkit "Epic Games" only, sound design on the revision-3 timeline.

  python3 mix3.py out.wav
"""
import os
import subprocess
import sys

import numpy as np
import scipy.io.wavfile as wf

from mix import DUR, ROOT, SR, load, put, sfx, synth_sub

ALT_OFFSET = 11.28   # Epic Games' post-break drop lands on 0:23.0 (LET THE GAME BEGIN)


def design():
    bus = np.zeros((int(DUR * SR), 2), np.float32)
    W1, W2, WH = sfx('1491'), sfx('1489'), sfx('2596')
    SLAM, BIG, DEEP, THUD = sfx('2150'), sfx('2908'), sfx('788'), sfx('2182')
    CLICK, CLICK2, CHIME, SOFT = sfx('900'), sfx('1113'), sfx('1108'), sfx('3109')
    SUB = synth_sub()
    put(bus, sfx('2900'), 0.0, 0.55, src_from=3.1, length=1.4)            # tension under the hook
    for t, a, g in [(3.58, WH, 0.8), (4.72, W1, 0.55), (5.86, W1, 0.55), (8.15, W2, 0.7), (8.72, W1, 0.55),
                    (9.86, W1, 0.5), (11.0, W1, 0.4), (12.15, W1, 0.4), (12.72, W2, 0.65), (15.0, W1, 0.6),
                    (15.86, W1, 0.5), (16.0, WH, 0.7), (17.29, W2, 0.65), (20.72, WH, 0.8), (22.95, WH, 0.85),
                    (24.15, W1, 0.6), (24.72, W1, 0.6), (25.29, W1, 0.6), (25.86, W2, 0.6), (27.0, W1, 0.6),
                    (27.57, W1, 0.55), (28.72, WH, 0.8), (29.86, W2, 0.6)]:
        put(bus, a, t, g, align='peak')
    for t, g in [(1.29, 0.9), (3.62, 0.7), (8.2, 0.6), (12.76, 0.6), (15.04, 0.6), (17.33, 0.7), (20.76, 0.75),
                 (25.9, 0.7)]:
        put(bus, SLAM, t, g, align='peak')
        put(bus, SUB, t, 0.35 * g)
    put(bus, BIG, 1.29, 0.6, align='peak')
    put(bus, sfx('2182'), 2.15, 0.7, align='peak')
    put(bus, sfx('2182'), 2.43, 0.7, align='peak', r=0.85)
    # taps on the real screens
    put(bus, CLICK, 5.6, 0.6)
    put(bus, CLICK2, 8.13, 0.9)
    put(bus, SLAM, 8.15, 0.7, align='peak')
    put(bus, BIG, 8.15, 0.45, align='peak')
    # the three game logos land
    for t, r in ((9.86, 0.85), (11.0, 1.0), (12.15, 1.1)):
        put(bus, THUD, t, 0.7, align='peak', r=r)
        put(bus, SUB, t, 0.25)
    # room code characters
    for i in range(9):
        put(bus, SOFT, 13.27 + i * 0.1425, 0.28, r=1.0 + i * 0.06)
    # the code reaches each friend
    for t, r in ((16.43, 1.0), (16.72, 1.12), (17.0, 1.26)):
        put(bus, CHIME, t, 0.55, r=r)
    # THEY JOIN: a chime per player, energy building to the fourth seat
    for t, r, g in ((17.29, 0.9, 0.4), (17.86, 1.0, 0.6), (19.01, 1.12, 0.65), (20.15, 1.33, 0.85)):
        put(bus, CHIME, t, g, r=r)
        put(bus, THUD, t, 0.4 * g, align='peak')
    put(bus, SLAM, 20.15, 0.5, align='peak')
    put(bus, sfx('2900'), 19.3, 0.35, src_from=3.6, length=1.4)
    # LET THE GAME BEGIN peak + piece impacts on every gameplay cut
    put(bus, DEEP, 23.0, 0.9, align='peak')
    put(bus, BIG, 23.0, 0.8, align='peak')
    put(bus, SLAM, 23.0, 0.9, align='peak')
    put(bus, SUB, 23.0, 0.7)
    for t in (23.4, 24.15, 24.72, 25.29, 27.0):
        put(bus, THUD, t, 0.9, align='peak')
        put(bus, SUB, t, 0.3)
    put(bus, sfx('368'), 27.2, 0.4, src_from=0.4, length=4.5)            # friends' energy
    put(bus, sfx('364'), 28.6, 0.22, src_from=4.0, length=4.0)
    put(bus, BIG, 28.72, 0.9, align='peak')                               # TRY IT NOW
    put(bus, SLAM, 28.72, 0.9, align='peak')
    put(bus, SUB, 28.72, 0.7)
    put(bus, BIG, 29.86, 0.8, align='peak')                               # logo reveal resolves
    put(bus, SLAM, 29.86, 0.7, align='peak')
    put(bus, SUB, 29.86, 0.6)
    put(bus, CHIME, 29.92, 0.4, r=0.8)
    put(bus, SLAM, 32.43, 0.35, align='peak')
    for t in (3.58, 8.15, 12.72, 15.0, 17.29, 20.72, 25.86):             # drum weight on scene changes
        put(bus, SUB, t, 0.35)
        put(bus, THUD, t, 0.45, align='peak')
    return bus


def music():
    bus = np.zeros((int(DUR * SR), 2), np.float32)
    put(bus, load(os.path.join(ROOT, 'music', '76.mp3')), 0.0, 0.85, src_from=ALT_OFFSET)
    k = int(0.3 * SR)
    bus[:k] *= np.linspace(0, 1, k)[:, None]
    k = int(1.6 * SR)
    bus[-k:] *= np.linspace(1, 0, k)[:, None] ** 1.3
    return bus


if __name__ == '__main__':
    out = sys.argv[1]
    mix = music() + design() * 0.8
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    tmp = out + '.raw.wav'
    wf.write(tmp, SR, mix.astype(np.float32))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-af', 'loudnorm=I=-14:TP=-1.0:LRA=11', '-ar', str(SR),
                    '-c:a', 'pcm_s16le', out], check=True)
    os.remove(tmp)
    print('wrote', out)
