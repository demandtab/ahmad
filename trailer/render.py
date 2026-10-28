"""Render the trailer.

  SC=0.25 python3 render.py sheet 0.5 1.2 ...   -> contact sheet of those times
  SC=1    python3 render.py video out.mp4 [workers]
"""
import os
import subprocess
import sys
from multiprocessing import Process

import cv2
import numpy as np

import common
import importlib

scenes = importlib.import_module(os.environ.get('SCENES', 'scenes'))

OUT = os.path.join(common.ROOT, 'out')
os.makedirs(OUT, exist_ok=True)


def sheet(times, path, cols=6):
    tiles = []
    for t in times:
        f = common.to_u8(scenes.render(t))
        f = cv2.resize(f, (270, 480), interpolation=cv2.INTER_AREA)
        cv2.putText(f, f'{t:.2f}', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(f)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def worker(i0, i1, path, crf):
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{common.W}x{common.H}',
           '-r', str(common.FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', '-x264-params', 'keyint=30', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(common.to_u8(scenes.render(i / common.FPS)).tobytes())
        if (i - i0) % 30 == 0:
            print(f'[{os.getpid()}] frame {i}/{i1}', flush=True)
    p.stdin.close()
    p.wait()


def video(out, workers=4, crf=14, t0=0.0, t1=scenes.DURATION):
    n0, n1 = int(round(t0 * common.FPS)), int(round(t1 * common.FPS))
    bounds = np.linspace(n0, n1, workers + 1).astype(int)
    parts = [os.path.join(OUT, f'part_{k}.mp4') for k in range(workers)]
    procs = [Process(target=worker, args=(bounds[k], bounds[k + 1], parts[k], crf)) for k in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    lst = os.path.join(OUT, 'parts.txt')
    with open(lst, 'w') as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', out],
                   check=True)


if __name__ == '__main__':
    if sys.argv[1] == 'sheet':
        sheet([float(x) for x in sys.argv[3:]], sys.argv[2])
    else:
        args = sys.argv[2:]
        video(args[0], int(args[1]) if len(args) > 1 else 4, int(args[2]) if len(args) > 2 else 14,
              float(args[3]) if len(args) > 3 else 0.0, float(args[4]) if len(args) > 4 else scenes.DURATION)
