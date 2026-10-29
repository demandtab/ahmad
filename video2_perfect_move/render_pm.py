"""Render "The Perfect Move".

  SC=0.3333 python3 render_pm.py sheet out.jpg 0.5 3.2 ...      # frames at those times
  SC=0.3333 python3 render_pm.py video out.mp4 [workers] [crf]   # picture only (mix_pm.py adds sound)
"""
import os
import subprocess
import sys
from multiprocessing import Process

import cv2
import numpy as np

import scenes_pm as S
import common as C


def sheet(times, path, cols=6):
    tiles = []
    for t in times:
        f = C.to_u8(S.render(t))
        f = cv2.resize(f, (360, 640), interpolation=cv2.INTER_AREA)
        cv2.putText(f, f'{t:.2f}', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(f)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def worker(i0, i1, path, crf):
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{C.W}x{C.H}',
           '-r', str(C.FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(C.to_u8(S.render(i / C.FPS)).tobytes())
    p.stdin.close()
    p.wait()


def video(out, workers=8, crf=16):
    n = int(round(S.DURATION * C.FPS))
    bounds = np.linspace(0, n, workers + 1).astype(int)
    parts = [f'{out}.part{k}.mp4' for k in range(workers)]
    procs = [Process(target=worker, args=(bounds[k], bounds[k + 1], parts[k], crf)) for k in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    lst = out + '.parts.txt'
    with open(lst, 'w') as fh:
        fh.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', out], check=True)
    for p in parts + [lst]:
        os.remove(p)


if __name__ == '__main__':
    if sys.argv[1] == 'sheet':
        sheet([float(x) for x in sys.argv[3:]], sys.argv[2])
    else:
        a = sys.argv[2:]
        video(a[0], int(a[1]) if len(a) > 1 else 8, int(a[2]) if len(a) > 2 else 16)
