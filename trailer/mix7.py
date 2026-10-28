"""Soundtrack for the final mobile video (scenes7): Mixkit "Epic Games" only, hits on the exact picture frames.

  python3 mix7.py out.wav
"""
import os
import subprocess
import sys

import numpy as np
import scipy.io.wavfile as wf

os.environ.setdefault('SC', '0.3333')
import mix3  # noqa: E402
from mix import SR, sfx, synth_sub  # noqa: E402
import scenes7  # noqa: E402
import scenes6  # noqa: E402

_put = mix3.put
REMOVED = {23.4, 24.15, 25.29, 25.86, 25.9, 27.0, 27.57}   # cuts that no longer exist in the reworked ending


def snapped(bus, a, t, *args, **kw):
    if round(t, 2) in REMOVED:
        return
    return _put(bus, a, scenes6.nom2out(t), *args, **kw)


if __name__ == '__main__':
    out = sys.argv[1]
    mix3.put = snapped
    design = mix3.design()
    snapped(design, sfx('1489'), 26.43, 0.6, align='peak')     # dissolve into the board tilt
    snapped(design, sfx('2182'), 26.43, 0.6, align='peak')
    snapped(design, synth_sub(), 26.43, 0.3)
    snapped(design, sfx('1491'), 24.72, 0.45, align='peak')
    mix3.put = _put
    mix = mix3.music() + design * 0.8
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    tmp = out + '.raw.wav'
    wf.write(tmp, SR, mix.astype(np.float32))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-af', 'loudnorm=I=-14:TP=-1.0:LRA=11', '-ar', str(SR),
                    '-c:a', 'pcm_s16le', out], check=True)
    os.remove(tmp)
    print('wrote', out)
