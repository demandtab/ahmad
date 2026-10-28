"""Soundtrack: music bed + sound design, synced to the picture timeline.

  python3 mix.py out.wav     # Mixkit "Epic Games" (free Mixkit licence) as the only score
"""
import functools
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000
DUR = 34.0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('SC', '0.25')
import scenes  # noqa: E402  (event times come from the picture timeline)


@functools.lru_cache(maxsize=None)
def load(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def sfx(name):
    return load(os.path.join(ROOT, 'sfx', f'{name}.mp3'))


def peak_offset(a):
    env = np.abs(a).max(1)
    hop = 480
    e = np.array([env[i:i + hop].max() for i in range(0, len(env), hop)])
    return int(e.argmax() * hop)


def rate(a, r):
    if r == 1.0:
        return a
    n = int(len(a) / r)
    x = np.linspace(0, len(a) - 1, n)
    return np.stack([np.interp(x, np.arange(len(a)), a[:, c]) for c in range(2)], 1).astype(np.float32)


def put(bus, a, t, gain=1.0, align='start', r=1.0, fade_out=None, src_from=0.0, length=None):
    a = rate(a, r)
    if src_from:
        a = a[int(src_from * SR):]
    if length:
        a = a[:int(length * SR)].copy()
        k = min(len(a), int(0.08 * SR))
        a[-k:] *= np.linspace(1, 0, k)[:, None]
    off = peak_offset(a) if align == 'peak' else 0
    i0 = int(t * SR) - off
    if i0 < 0:
        a = a[-i0:]
        i0 = 0
    a = a[:max(0, len(bus) - i0)] * gain
    if fade_out:
        k = min(len(a), int(fade_out * SR))
        a[-k:] *= np.linspace(1, 0, k)[:, None]
    bus[i0:i0 + len(a)] += a


def synth_sub(t_len=1.2, f0=55):
    n = int(t_len * SR)
    tt = np.arange(n) / SR
    f = f0 * np.exp(-tt * 1.5)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-tt * 3.0)
    return np.stack([s, s], 1).astype(np.float32)


def design():
    bus = np.zeros((int(DUR * SR), 2), np.float32)

    def pt(bus, a, t, *args, **kw):  # event times are on the original edit clock
        put(bus, a, scenes.old2new(t), *args, **kw)
    W1, W2, WH = sfx('1491'), sfx('1489'), sfx('2596')
    SLAM, BIG, DEEP, THUD = sfx('2150'), sfx('2908'), sfx('788'), sfx('2182')
    CLICK, CLICK2, CHIME, SOFT = sfx('900'), sfx('1113'), sfx('1108'), sfx('3109')
    SUB = synth_sub()

    # 0:00 tension riser into the hook title
    pt(bus, sfx('2900'), 0.0, 0.55, src_from=3.3, length=1.75)
    for t, a, g in [(2.95, WH, 0.8), (6.95, W1, 0.7), (7.75, W1, 0.55), (8.4, W1, 0.55), (11.62, W2, 0.7),
                    (14.95, W1, 0.45), (scenes.BURST_T, WH, 0.75), (17.95, W2, 0.7), (18.88, W1, 0.6),
                    (20.95, W2, 0.65), (22.95, WH, 0.85), (23.82, W1, 0.6), (24.58, W1, 0.6), (25.33, W1, 0.6),
                    (25.97, W2, 0.6), (27.38, W1, 0.6), (27.88, W1, 0.55), (29.27, WH, 0.8), (29.95, W2, 0.6)]:
        pt(bus, a, t, g, align='peak')
    pt(bus, W2, 30.6, 0.45, align='peak', r=0.7)  # logo particles swirling in
    # title slams
    for t, g in [(0.95, 0.9), (3.05, 0.7), (7.05, 0.7), (12.05, 0.6), (15.05, 0.6), (18.05, 0.7),
                 (21.05, 0.7), (26.05, 0.7), (31.6, 0.6)]:
        pt(bus, SLAM, t, g, align='peak')
        pt(bus, SUB, t, 0.35 * g)
    pt(bus, BIG, 0.95, 0.6, align='peak')
    # the two new armies light up
    pt(bus, THUD, 1.72, 0.7, align='peak')
    pt(bus, THUD, 1.92, 0.7, align='peak', r=0.85)
    # game panels ignite: CheQuers, BaQugammon, then Quess biggest
    for t, g, r in [(4.2, 0.5, 1.1), (5.2, 0.55, 1.0), (6.0, 0.9, 0.85)]:
        pt(bus, W1, t, g * 0.8, align='peak', r=r)
        pt(bus, THUD, t, g, align='peak', r=r)
    pt(bus, SUB, 6.0, 0.4)
    # taps from the real recording
    pt(bus, CLICK, 5.85, 0.6)
    pt(bus, CLICK, 7.7, 0.6)
    pt(bus, CLICK, 8.41, 0.6)
    pt(bus, CLICK2, scenes.CLICK_T - 0.02, 0.9)
    pt(bus, SLAM, scenes.CLICK_T, 0.7, align='peak')
    pt(bus, BIG, scenes.CLICK_T, 0.45, align='peak')
    # room code digits materialise with a soft chime each
    for i, t in enumerate(scenes.char_arrivals()):
        pt(bus, SOFT, t - 0.02, 0.28, r=1.0 + i * 0.06)
    # particles reach each friend
    for a, r in zip(scenes.AVATARS, (1.0, 1.12, 1.26)):
        pt(bus, CHIME, a['t'], 0.55, r=r)
    # players join, the fourth in slow motion
    for t, r, g in zip(scenes.JOIN_T, (0.9, 1.0, 1.12, 1.33), (0.45, 0.6, 0.6, 0.8)):
        pt(bus, CHIME, t, g, r=r)
    pt(bus, SLAM, scenes.JOIN_T[-1], 0.4, align='peak')
    # LET THE GAME BEGIN peak + heavy piece impacts on each gameplay cut
    pt(bus, DEEP, 23.0, 0.9, align='peak')
    pt(bus, BIG, 23.0, 0.8, align='peak')
    pt(bus, SLAM, 23.0, 0.9, align='peak')
    pt(bus, SUB, 23.0, 0.7)
    for t in (23.4, 23.85, 24.6, 25.35, 27.4):
        pt(bus, THUD, t, 0.9, align='peak')
        pt(bus, SUB, t, 0.3)
    # friends: warm laughter under the couch shots, then the crowd lifts the ending
    crowd = sfx('368')
    pt(bus, crowd, 27.6, 0.42, src_from=0.4, length=6.3)
    pt(bus, sfx('364'), 29.2, 0.25, src_from=4.0, length=4.7)
    # TRY IT NOW final hit with screen shake
    pt(bus, BIG, 29.3, 0.9, align='peak')
    pt(bus, SLAM, 29.3, 0.9, align='peak')
    pt(bus, SUB, 29.3, 0.7)
    # logo sting
    pt(bus, BIG, scenes.LOGO_T, 0.8, align='peak')
    pt(bus, SLAM, scenes.LOGO_T, 0.8, align='peak')
    pt(bus, SUB, scenes.LOGO_T, 0.6)
    pt(bus, CHIME, scenes.LOGO_T + 0.05, 0.4, r=0.8)
    # extra drum weight on every beat-aligned scene change
    for t in (3.58, 7.01, 8.15, 11.58, 15.0, 18.44, 20.72, 25.86, 29.86):
        put(bus, SUB, t, 0.35)
        put(bus, THUD, t, 0.45, align='peak')
    return bus


def music():
    """Mixkit "Epic Games" only, offset so its post-break drop lands on 0:23.0."""
    bus = np.zeros((int(DUR * SR), 2), np.float32)
    m = load(os.path.join(ROOT, 'music', '76.mp3'))
    put(bus, m, 0.0, 0.85, src_from=ALT_OFFSET)
    k = int(0.3 * SR)
    bus[:k] *= np.linspace(0, 1, k)[:, None]
    k = int(1.6 * SR)  # resolve under the logo
    bus[-k:] *= np.linspace(1, 0, k)[:, None] ** 1.3
    return bus


ALT_OFFSET = float(os.environ.get('ALT_OFFSET', '11.28'))

if __name__ == '__main__':
    out = sys.argv[1]
    mix = music() + design() * 0.8
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    tmp = out + '.raw.wav'
    import scipy.io.wavfile as wf
    wf.write(tmp, SR, mix.astype(np.float32))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-af', 'loudnorm=I=-14:TP=-1.0:LRA=11',
                    '-ar', str(SR), '-c:a', 'pcm_s16le', out], check=True)
    os.remove(tmp)
    print('wrote', out)
