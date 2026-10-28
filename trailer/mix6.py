"""Soundtrack for the final videos: Mixkit "Epic Games" only; every sound-design hit on the exact picture frame.

  python3 mix6.py out.wav
"""
import os
import subprocess
import sys

import numpy as np
import scipy.io.wavfile as wf

import mix3
from mix import SR

os.environ.setdefault('SC', '0.3333')
import scenes6  # noqa: E402

_put = mix3.put


def snapped(bus, a, t, *args, **kw):
    return _put(bus, a, scenes6.nom2out(t), *args, **kw)


if __name__ == '__main__':
    out = sys.argv[1]
    mix3.put = snapped
    design = mix3.design()
    mix3.put = _put
    mix = mix3.music() + design * 0.8
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    tmp = out + '.raw.wav'
    wf.write(tmp, SR, mix.astype(np.float32))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-af', 'loudnorm=I=-14:TP=-1.0:LRA=11', '-ar', str(SR),
                    '-c:a', 'pcm_s16le', out], check=True)
    os.remove(tmp)
    print('wrote', out)
