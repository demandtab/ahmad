"""Render one variation in 4 parallel parts, re-render any part that comes up short, then concat."""
import os
import subprocess
import sys
from multiprocessing import Process

import numpy as np

import common
import render
import importlib

scenes = importlib.import_module(os.environ.get('SCENES', 'scenes'))

var = os.environ.get('OUTDIR', os.environ['VARIANT'])
out_dir = os.path.join(common.ROOT, 'out', var)
os.makedirs(out_dir, exist_ok=True)
n = int(round(scenes.DURATION * common.FPS))
bounds = np.linspace(0, n, 5).astype(int)
parts = [os.path.join(out_dir, f'part_{k}.mp4') for k in range(4)]


def frames(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-show_entries', 'stream=nb_read_packets',
                        '-of', 'csv=p=0', p], capture_output=True, text=True)
    return int(r.stdout.strip() or 0) if r.returncode == 0 else 0


for attempt in range(3):
    todo = [k for k in range(4) if frames(parts[k]) != bounds[k + 1] - bounds[k]]
    if not todo:
        break
    print(var, 'rendering parts', todo, flush=True)
    procs = [Process(target=render.worker, args=(bounds[k], bounds[k + 1], parts[k], 17)) for k in todo]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
lst = os.path.join(out_dir, 'parts.txt')
with open(lst, 'w') as fh:
    fh.writelines(f"file '{p}'\n" for p in parts)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy',
                os.path.join(out_dir, 'video.mp4')], check=True)
print(var, 'done', frames(os.path.join(out_dir, 'video.mp4')), 'frames', flush=True)
