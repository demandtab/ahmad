"""Soundtrack for "The Perfect Move": Mixkit "Silent Descent" (#614) bed + sound design on the beat grid.

  python3 mix_pm.py out.wav

Reuses trailer/mix.py (loader, peak alignment, synthesised sub-drop). Arc: calm (1-3), unease (4-5),
riser (6), 0.25 s stop-down, impact + sub on line 7 (the biggest hit), full energy into the logo (loudest).
"""
import os
import subprocess
import sys

import numpy as np
import scipy.io.wavfile as wf

os.environ.setdefault('SC', '0.25')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'trailer'))
from mix import SR, load, sfx, put, synth_sub  # noqa: E402
import scenes_pm as S  # noqa: E402

DUR = S.DURATION
K = S.K


def db(x):
    return 10 ** (x / 20)


def env(points, n):
    """Piecewise-linear gain envelope over [(t, dB), ...]."""
    t = np.arange(n) / SR
    xs, ys = zip(*points)
    return db(np.interp(t, xs, ys))[:, None].astype(np.float32)


def music():
    n = int(DUR * SR)
    bus = np.zeros((n, 2), np.float32)
    m = load(os.path.join(S.C.ROOT, 'music', '614.mp3'))
    put(bus, m, 0.0, 1.0, src_from=S.MUSIC_FILE_OFFSET + S.SHIFT)
    hit, stop = K['hit'], S.STOP
    g = env([(0, -60), (0.35, -20), (K['p2'], -20), (K['p2'] + 0.5, -19), (K['p3'], -19),
             (stop - 0.4, -18), (stop, -44), (hit - 0.01, -46),           # 0.25 s near-silence
             (hit, -22), (K['react'], -19), (K['l8'] - 0.05, -18),         # comes back under the hit
             (K['l8'], -8), (K['outro'] - 0.1, -7), (K['outro'] + 0.4, -4), (DUR - 1.0, -4), (DUR, -30)], n)   # payoff = loudest
    return bus * g


def design():
    n = int(DUR * SR)
    bus = np.zeros((n, 2), np.float32)
    snap = S.S
    # whoosh on every caption (peak on the text hit, which is on the beat)
    wh = ['1489', '2596', '1491']
    for i, (t0, t1, kind, lines, fx) in enumerate(S.TEXT):
        if fx == 'hard':
            continue
        g = 0.32 if kind == 'narr' else 0.24
        if t0 >= K['l8']:
            g = 0.42
        put(bus, sfx(wh[i % 3]), t0, g, align='peak')
    # soft wooden clicks on the real move frames (selection + landing)
    p1m, p1a = K['p1m'], K['p1a']
    d_m = p1a - p1m
    land1 = p1m + 1.0 + (5.1 - 3.75) * (d_m - 1.0) / 1.85
    for t, g in [(K['p1'] + 0.12, 0.35), (land1, 0.55), (K['p2'] + 0.12, 0.35), (K['p2a'] - 0.04, 0.5)]:
        put(bus, sfx('1113'), t, g, align='peak')
        put(bus, sfx('900'), t, g * 0.5, align='peak')
    # low boom on each army flash (turn starts are cuts on the beat)
    for t, _ in S.TURNS:
        put(bus, sfx('2182'), t, 0.38, align='peak', r=0.8)
    # riser into line 7, peaking at the stop-down; then 0.25 s near-silence; then the biggest hit
    put(bus, sfx('2900'), S.STOP, 0.32, align='peak', fade_out=0.05)
    put(bus, sfx('2150'), K['hit'], 1.0, align='peak')
    put(bus, sfx('788'), K['hit'], 0.8, align='peak')
    put(bus, synth_sub(1.6, 52), K['hit'], 0.9)
    put(bus, sfx('1491'), K['hit'] + 0.86, 0.3, align='peak')          # "That changes everything." lands
    # hits on every montage cut
    for t in S.MONTAGE_HITS:
        put(bus, sfx('2908'), t, 0.75, align='peak')
        put(bus, sfx('2182'), t, 0.5, align='peak')
    put(bus, synth_sub(1.2, 58), K['l8'], 0.6)
    # into the logo: impact on the outro cut, sting on PLAY QUESS (second peak)
    put(bus, sfx('2182'), K['outro'], 0.55, align='peak')
    pq = S.TEXT[-1][0]
    put(bus, sfx('2150'), pq, 0.85, align='peak')
    put(bus, sfx('3109'), pq, 0.55, align='peak')
    put(bus, synth_sub(1.4, 55), pq, 0.7)
    for t in (K['outro'] + 0.9, K['outro'] + 3.2):                     # keep the end card driving into the logo
        put(bus, sfx('2908'), t, 0.6, align='peak')
    i0, i1 = int(S.STOP * SR), int(K['hit'] * SR) - int(0.012 * SR)
    bus[i0:i1] *= 0.04                                                    # the 0.25 s stop-down stays near-silent
    bus[int(K['p3'] * SR):i0] *= 0.72                                      # tension sits under the payoff
    bus[int(K['hit'] * SR) + int(0.5 * SR):int(K['l8'] * SR)] *= 0.7       # hit tail; payoff stays the loudest part
    return bus


if __name__ == '__main__':
    out = sys.argv[1]
    mix = music() + design() * 0.8
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    tmp = out + '.raw.wav'
    wf.write(tmp, SR, mix.astype(np.float32))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-ar', str(SR),
                    '-c:a', 'pcm_s16le', out], check=True)
    os.remove(tmp)
    print('wrote', out)
