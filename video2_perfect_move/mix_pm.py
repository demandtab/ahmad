"""Soundtrack for "The Perfect Move" to the v2-fix [AUDIO] map: ONE music track + exactly THREE sound effects.

  python3 mix_pm.py out.wav                      # placeholder bed: Mixkit "Silent Descent" (#614), lift on 21.0
  python3 mix_pm.py out.wav track.wav [start_s]  # the client's generated 34 s track (no levelling, map only)

Level map (dB relative to full level): 0-3 -6 | 3-11 slow rise to 0 | 11-21 0 | 21-29 0 (climax, hits 21.0 and 26.0
in the track) | 29-32 -6 under the end card | 32-33.5 fade to silence. SFX: 11.0 tick, 21.0 capture hit, 29.0 chime.
Final: -14 LUFS integrated, true peak <= -1.5 dBTP (two-pass loudnorm, linear).
"""
import json
import os
import subprocess
import sys

import numpy as np
import scipy.io.wavfile as wf

os.environ.setdefault('SC', '0.25')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'trailer'))
from mix import SR, load, sfx, put  # noqa: E402
import scenes_pm as S  # noqa: E402

DUR = S.DURATION   # 34.0


def db(x):
    return 10 ** (x / 20)


def env(points, n):
    t = np.arange(n) / SR
    xs, ys = zip(*points)
    return db(np.interp(t, xs, ys))[:, None].astype(np.float32)


def level(bus, target_db=-18.0, win=1.5, max_gain_db=10.0):
    """Placeholder only: ride the Mixkit bed to a steady level so the map, not the track, shapes it."""
    n = len(bus)
    hop = int(0.05 * SR)
    m = bus.mean(1)
    rms = np.array([np.sqrt((m[i:i + hop] ** 2).mean() + 1e-12) for i in range(0, n, hop)])
    k = max(1, int(win / 0.05))
    rms = np.convolve(rms, np.ones(k) / k, mode='same')
    g = np.clip(db(target_db) / rms, db(-12), db(max_gain_db))
    return bus * np.interp(np.arange(n), np.arange(len(g)) * hop, g)[:, None].astype(np.float32)


MAP = [(0.0, -6), (3.0, -6), (11.0, 0), (21.0, 0), (29.0, 0), (29.25, -6), (32.0, -6), (33.5, -90), (34.0, -90)]


def music(track=None, start=0.0):
    n = int(DUR * SR)
    bus = np.zeros((n, 2), np.float32)
    if track:
        put(bus, load(track), 0.0, 1.0, src_from=start)
    else:
        put(bus, load(os.path.join(S.C.ROOT, 'music', '614.mp3')), 0.0, 1.0, src_from=S.MUSIC_FILE_OFFSET + S.SHIFT)
    k = int(0.02 * SR)
    bus[:k] *= np.linspace(0, 1, k)[:, None]
    return bus * env(MAP, n)


def design():
    n = int(DUR * SR)
    bus = np.zeros((n, 2), np.float32)
    put(bus, sfx('1113'), 11.0, 0.30, align='peak')    # soft piece-placement tick
    put(bus, sfx('2150'), 21.0, 0.32, align='peak')    # capture hit on the drum hit, under the music
    put(bus, sfx('3109'), 29.0, 0.30, align='peak')    # soft logo chime
    return bus


def finish(mix, out):
    tmp = out + '.raw.wav'
    wf.write(tmp, SR, mix.astype(np.float32))
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', tmp, '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json',
                        '-f', 'null', '-'], capture_output=True, text=True).stderr
    m = json.loads(r[r.rindex('{'):r.rindex('}') + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-af', af, '-ar', str(SR), '-c:a', 'pcm_s16le', out],
                   check=True)
    os.remove(tmp)


if __name__ == '__main__':
    out = sys.argv[1]
    track = sys.argv[2] if len(sys.argv) > 2 else None
    mix = music(track, float(sys.argv[3]) if len(sys.argv) > 3 else 0.0) + design()
    finish(mix, out)
    print('wrote', out)
