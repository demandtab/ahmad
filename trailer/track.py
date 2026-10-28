"""Track the green-screen monitor in a stock clip: per-frame screen quad (TL, TR, BR, BL) in source pixels.

  python3 track.py pc51602            -> src/pc51602/track.json
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (40, 110, 105), (85, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8) * 255


def edge_line(pts):
    """Robust line fit (vx, vy, x0, y0) through boundary points."""
    return cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()


def intersect(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    b = np.array([x2 - x1, y2 - y1])
    s = np.linalg.solve(A, b)[0]
    return np.array([x1 + s * vx1, y1 + s * vy1])


def quad_from_mask(m):
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(c).reshape(-1, 2)
    s, d = hull.sum(1), hull[:, 0] - hull[:, 1]
    tl, br = hull[np.argmin(s)], hull[np.argmax(s)]
    tr, bl = hull[np.argmax(d)], hull[np.argmin(d)]
    rough = np.array([tl, tr, br, bl])
    # refine each corner by intersecting robust fits of the four edges (contour points near each edge)
    lines = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        p, q = rough[a], rough[b]
        v = q - p
        L = np.linalg.norm(v) + 1e-6
        n = np.array([-v[1], v[0]]) / L
        t = ((c - p) @ v) / (L * L)
        dist = np.abs((c - p) @ n)
        sel = c[(t > 0.12) & (t < 0.88) & (dist < 12)]
        lines.append(edge_line(sel) if len(sel) > 20 else None)
    out = rough.copy()
    for i, (la, lb) in enumerate(((3, 0), (0, 1), (1, 2), (2, 3))):
        if lines[la] is not None and lines[lb] is not None:
            out[i] = intersect(lines[la], lines[lb])
    return out


def main(name):
    d = os.path.join(ROOT, 'src', name)
    files = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    quads = []
    for f in files:
        img = cv2.imread(os.path.join(d, f))
        m = green_mask(img)
        quads.append(quad_from_mask(m).tolist() if m is not None else None)
    Q = np.array([q if q is not None else [[np.nan] * 2] * 4 for q in quads], np.float64)
    # fill gaps and smooth (centered moving average over 7 frames)
    for k in range(4):
        for j in range(2):
            v = Q[:, k, j]
            ok = ~np.isnan(v)
            v[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], v[ok])
            pad = np.pad(v, 3, mode='edge')
            Q[:, k, j] = np.convolve(pad, np.ones(7) / 7, mode='valid')
    json.dump(Q.tolist(), open(os.path.join(d, 'track.json'), 'w'))
    print(name, len(files), 'frames; first quad', np.round(Q[0]).tolist(), 'last', np.round(Q[-1]).tolist())




def dominance_mask(img, ratio=1.35, gmin=45):
    f = img.astype(np.int32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    m = ((g > ratio * np.maximum(r, b)) & (g > gmin)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8) * 255


def partial_quad(m, aspect_h=0.62):
    """Screen cut off at the top: fit left/right/bottom edges, extend the sides upward."""
    H_, W_ = m.shape
    ys, xs = np.nonzero(m)
    rows = np.unique(ys)
    left, right = [], []
    for y in rows[::6]:
        xr = xs[ys == y]
        if len(xr) > 50:
            left.append((xr.min(), y))
            right.append((xr.max(), y))
    left, right = np.array(left, np.float32), np.array(right, np.float32)
    ymax = ys.max()
    # sides: use the upper 60% of rows (below that the mic may cut in)
    yl = left[:, 1] < ys.min() + 0.6 * (ymax - ys.min())
    L = edge_line(left[yl])
    Rl = edge_line(right[yl])
    cols = np.unique(xs)
    bottom = np.array([(x, ys[xs == x].max()) for x in cols[::6]], np.float32)
    # drop the mic-shadowed stretch: keep points within 25px of the max-bottom trend
    med = np.median(bottom[:, 1])
    Bl = edge_line(bottom[bottom[:, 1] > med - 25])
    bl, br = intersect(L, Bl), intersect(Rl, Bl)
    width = np.linalg.norm(br - bl)
    def up(line, p):
        vx, vy = line[0], line[1]
        if vy > 0:
            vx, vy = -vx, -vy
        return p + np.array([vx, vy]) * width * aspect_h
    return np.array([up(L, bl), up(Rl, br), br, bl])


def main_partial(name):
    d = os.path.join(ROOT, 'src', name)
    files = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    Q = np.array([partial_quad(dominance_mask(cv2.imread(os.path.join(d, f)))) for f in files], np.float64)
    for k in range(4):
        for j in range(2):
            pad = np.pad(Q[:, k, j], 4, mode='edge')
            Q[:, k, j] = np.convolve(pad, np.ones(9) / 9, mode='valid')
    json.dump(Q.tolist(), open(os.path.join(d, 'track.json'), 'w'))
    print(name, len(files), np.round(Q[0]).tolist())


if __name__ == '__main__':
    (main_partial if len(sys.argv) > 2 and sys.argv[2] == 'partial' else main)(sys.argv[1])
