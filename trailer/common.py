"""Compositing engine for the Quess vertical trailer.

All layout numbers are written in final 4K units (2160x3840) and scaled with P()
so the same code renders fast low-res previews (env SC=0.25) and the 4K master.
Frames are float32 BGR in [0, 1+] (values above 1 feed the bloom).
"""
import functools
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
FONTS = os.path.join(ROOT, 'fonts')
SC = float(os.environ.get('SC', '1.0'))
W, H = int(round(2160 * SC)), int(round(3840 * SC))
FPS = 30
cv2.setNumThreads(1)


def P(v):
    return v * SC


# ---------------------------------------------------------------- easing
def clamp01(x):
    return max(0.0, min(1.0, x))


def seg(t, a, b):
    return clamp01((t - a) / (b - a)) if b > a else float(t >= b)


def eo3(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ei3(x):
    x = clamp01(x)
    return x ** 3


def eio(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a + (b - a) * x


def piecewise(t, keys):
    """Linear interpolation through [(t, v), ...] with eased segments."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return lerp(v0, v1, eio((t - t0) / (t1 - t0)))
    return keys[-1][1]


def remap(t, keys):
    """Pure linear time remap (used for speed ramps)."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return lerp(v0, v1, (t - t0) / (t1 - t0))
    return keys[-1][1]


# ---------------------------------------------------------------- sources
class Clip:
    def __init__(self, name, fps):
        self.dir = os.path.join(SRC, name)
        self.fps = fps
        self.n = len([f for f in os.listdir(self.dir) if f.endswith('.jpg')])

    @functools.lru_cache(maxsize=4)
    def frame(self, i):
        i = int(min(max(i, 0), self.n - 1))
        img = cv2.imread(os.path.join(self.dir, f'{i + 1:04d}.jpg'))
        if SC < 0.99:
            img = cv2.resize(img, None, fx=max(SC * 1.5, 0.25), fy=max(SC * 1.5, 0.25),
                             interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def at(self, t):
        return self.frame(int(round(t * self.fps)))


CLIPS = {}


def clip(name, fps=30):
    if name not in CLIPS:
        CLIPS[name] = Clip(name, fps)
    return CLIPS[name]


@functools.lru_cache(maxsize=None)
def still(name):
    p = os.path.join(SRC, 'stills', name)
    return cv2.imread(p, cv2.IMREAD_UNCHANGED)


# ---------------------------------------------------------------- 2D placement
def affine_cover(w, h, focus, zoom, angle, out, dx=0.0, dy=0.0, mode='cover'):
    ow, oh = out
    base = max(ow / w, oh / h) if mode == 'cover' else ow / w
    s = base * zoom
    fx, fy = focus[0] * w, focus[1] * h
    a = math.radians(angle)
    c, sn = math.cos(a) * s, math.sin(a) * s
    M = np.array([[c, -sn, 0], [sn, c, 0]], np.float32)
    M[:, 2] = [ow / 2 + dx - (c * fx - sn * fy), oh / 2 + dy - (sn * fx + c * fy)]
    return M


def cover(img, focus=(0.5, 0.5), zoom=1.0, angle=0.0, out=None, dx=0.0, dy=0.0, mode='cover',
          border=cv2.BORDER_REFLECT):
    out = out or (W, H)
    h, w = img.shape[:2]
    M = affine_cover(w, h, focus, zoom, angle, out, dx, dy, mode)
    return cv2.warpAffine(img, M, out, flags=cv2.INTER_LINEAR, borderMode=border)


def band(img, focus=(0.5, 0.5), zoom=1.0, angle=0.0, dx=0.0, dy=0.0, bg_dim=0.33, feather=180):
    """Landscape footage in a vertical frame: sharp band over a blurred, darkened fill."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, w // 10), max(8, h // 10)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.2)
    bg = cover(small, focus=(0.5, 0.5), zoom=1.08 + 0.05 * zoom, angle=angle * 0.5) * bg_dim
    M = affine_cover(w, h, focus, zoom, angle, (W, H), dx, dy, mode='width')
    fg = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = cv2.warpAffine(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    k = max(1.0, P(feather) / 3)
    m = cv2.GaussianBlur(m, (0, 0), k)[..., None]
    return bg * (1 - m) + fg * m


def place(frame, img, cx, cy, s=1.0, alpha=1.0, angle=0.0, add=False):
    """Composite an RGBA (or RGB) float image centred at (cx, cy) in frame pixels."""
    if alpha <= 0.001 or s <= 0.001:
        return
    h, w = img.shape[:2]
    a = math.radians(angle)
    c, sn = math.cos(a) * s, math.sin(a) * s
    hw = (abs(c) * w + abs(sn) * h) / 2 + 2
    hh = (abs(sn) * w + abs(c) * h) / 2 + 2
    x0, x1 = int(max(0, math.floor(cx - hw))), int(min(frame.shape[1], math.ceil(cx + hw)))
    y0, y1 = int(max(0, math.floor(cy - hh))), int(min(frame.shape[0], math.ceil(cy + hh)))
    if x1 <= x0 or y1 <= y0:
        return
    M = np.array([[c, -sn, 0], [sn, c, 0]], np.float32)
    M[:, 2] = [cx - x0 - (c * w / 2 - sn * h / 2), cy - y0 - (sn * w / 2 + c * h / 2)]
    roi = cv2.warpAffine(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    dst = frame[y0:y1, x0:x1]
    if img.shape[2] == 4:
        al = roi[..., 3:4] * alpha
        rgb = roi[..., :3]
    else:
        al = alpha
        rgb = roi
    if add:
        dst += rgb * al
    else:
        dst[:] = dst * (1 - al) + rgb * al


def soft_blob(size, sigma_frac=0.22):
    g = cv2.getGaussianKernel(size, size * sigma_frac)
    k = (g @ g.T)
    return (k / k.max()).astype(np.float32)


def glow_add(frame, layer, strength, sigmas=(30, 110), down=8):
    """Additive multi-radius glow of `layer` (full-res float) onto frame."""
    small = cv2.resize(layer, (W // down, H // down), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for s in sigmas:
        acc += cv2.GaussianBlur(small, (0, 0), max(0.6, P(s) / down))
    frame += cv2.resize(acc, (W, H)) * strength


def zoom_blur(f, amount, cx=None, cy=None, n=6):
    if amount <= 0.002:
        return f
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    acc = f.copy()
    for i in range(1, n):
        s = 1 + amount * i / (n - 1)
        M = np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float32)
        acc += cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / n


def dir_blur(f, dx, dy, n=5):
    if abs(dx) + abs(dy) < 1:
        return f
    acc = f.copy()
    for i in range(1, n):
        k = i / (n - 1) - 0.5
        M = np.array([[1, 0, dx * k], [0, 1, dy * k]], np.float32)
        acc += cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / n


# ---------------------------------------------------------------- 3D device
def rotm(yaw, pitch, roll):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


F_PX = 2600.0


def proj(Pw):
    return np.stack([W / 2 + P(F_PX) * Pw[:, 0] / Pw[:, 2],
                     H / 2 + P(F_PX) * Pw[:, 1] / Pw[:, 2]], 1).astype(np.float32)


def rrect_mask(w, h, r):
    m = np.zeros((h * 2, w * 2), np.uint8)
    R = r * 2
    cv2.rectangle(m, (R, 0), (w * 2 - R - 1, h * 2 - 1), 255, -1)
    cv2.rectangle(m, (0, R), (w * 2 - 1, h * 2 - R - 1), 255, -1)
    for cx, cy in [(R, R), (w * 2 - R - 1, R), (R, h * 2 - R - 1), (w * 2 - R - 1, h * 2 - R - 1)]:
        cv2.circle(m, (cx, cy), R, 255, -1, cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


SRC_W, SRC_H = 1920, 1006  # recording frame after removing the window title bar / taskbar


class Device:
    """A landscape phone whose screen shows the real recording."""

    def __init__(self):
        self.SW, self.SH = int(P(2100)), int(P(2100) * SRC_H / SRC_W)
        self.BZ = int(P(46))
        self.TW, self.TH = self.SW + 2 * self.BZ, self.SH + 2 * self.BZ
        TW, TH, BZ = self.TW, self.TH, self.BZ
        body = rrect_mask(TW, TH, int(P(150)))
        inner = np.zeros((TH, TW), np.float32)
        inner[int(P(10)):TH - int(P(10)), int(P(10)):TW - int(P(10))] = \
            rrect_mask(TW - 2 * int(P(10)), TH - 2 * int(P(10)), int(P(140)))
        rim = np.clip(body - inner, 0, 1)
        grad = np.linspace(0.62, 0.22, TH, dtype=np.float32)[:, None]
        rgb = np.zeros((TH, TW, 3), np.float32)
        rgb[:] = (0.075, 0.07, 0.065)
        metal = np.repeat(grad, TW, 1)[..., None] * np.array([0.95, 0.97, 1.0], np.float32)
        rgb = rgb * (1 - rim[..., None]) + metal * rim[..., None]
        # camera punch hole on the left bezel (landscape)
        cv2.circle(rgb, (int(BZ * 0.5), TH // 2), max(1, int(P(11))), (0.02, 0.02, 0.025), -1, cv2.LINE_AA)
        cv2.circle(rgb, (int(BZ * 0.5) - max(1, int(P(3))), TH // 2 - max(1, int(P(3)))), max(1, int(P(3))),
                   (0.35, 0.3, 0.3), -1, cv2.LINE_AA)
        self.body_rgb = rgb
        self.body_a = body
        sm = np.zeros((TH, TW), np.float32)
        sm[BZ:BZ + self.SH, BZ:BZ + self.SW] = rrect_mask(self.SW, self.SH, int(P(80)))
        self.screen_mask = sm[..., None]
        # glass gloss: soft diagonal band
        yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
        d = (xx / TW * 0.8 + yy / TH * 0.6)
        self.gloss = (np.exp(-((d - 0.38) / 0.09) ** 2) * 0.07 + np.exp(-((d - 0.55) / 0.03) ** 2) * 0.035)[..., None] \
            * self.screen_mask
        self.dw = 2.0
        self.dh = self.dw * TH / TW
        self.th = 0.075

    def texture(self, screen):
        TW, TH, BZ = self.TW, self.TH, self.BZ
        scr = cv2.resize(screen, (self.SW, self.SH), interpolation=cv2.INTER_AREA if SC < 1 else cv2.INTER_LINEAR)
        full = np.zeros((TH, TW, 3), np.float32)
        full[BZ:BZ + self.SH, BZ:BZ + self.SW] = scr
        rgb = self.body_rgb * (1 - self.screen_mask) + full * self.screen_mask + self.gloss
        return np.dstack([rgb, self.body_a])

    def corners(self, pose, z_off=0.0, mirror_y=None):
        x, y, z, yaw, pitch, roll = pose
        R = rotm(yaw, pitch, roll)
        loc = np.array([[-self.dw / 2, -self.dh / 2, z_off], [self.dw / 2, -self.dh / 2, z_off],
                        [self.dw / 2, self.dh / 2, z_off], [-self.dw / 2, self.dh / 2, z_off]])
        Pw = (R @ loc.T).T + np.array([x, y, z])
        if mirror_y is not None:
            Pw[:, 1] = 2 * mirror_y - Pw[:, 1]
        return Pw

    def outline(self, pose, z_off=0.0):
        """Rounded-rectangle silhouette of the body in world space."""
        x, y, z, yaw, pitch, roll = pose
        r = self.dw * 150 / 2192
        hx, hy = self.dw / 2 - r, self.dh / 2 - r
        pts = []
        for cx, cy, a0 in ((hx, -hy, -90), (hx, hy, 0), (-hx, hy, 90), (-hx, -hy, 180)):
            for a in np.radians(np.linspace(a0, a0 + 90, 8)):
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z_off))
        loc = np.array(pts)
        return (rotm(yaw, pitch, roll) @ loc.T).T + np.array([x, y, z])

    def floor_y(self, pose, gap=0.2):
        Pw = np.vstack([self.corners(pose), self.corners(pose, self.th)])
        return Pw[:, 1].max() + gap

    def homography(self, pose, mirror_y=None):
        src = np.array([[0, 0], [self.TW, 0], [self.TW, self.TH], [0, self.TH]], np.float32)
        return cv2.getPerspectiveTransform(src, proj(self.corners(pose, mirror_y=mirror_y)))

    def map_screen(self, Hm, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 2)
        u = self.BZ + pts[:, 0] * self.SW / SRC_W
        v = self.BZ + pts[:, 1] * self.SH / SRC_H
        return cv2.perspectiveTransform(np.stack([u, v], 1)[None], Hm)[0]

    def aim(self, pose, src_pt, target):
        """Shift the device so a recording-space point lands on a frame-space target."""
        pose = list(pose)
        for _ in range(3):
            p = self.map_screen(self.homography(pose), [src_pt])[0]
            pose[0] += (target[0] - p[0]) * pose[2] / P(F_PX)
            pose[1] += (target[1] - p[1]) * pose[2] / P(F_PX)
        return pose


DEV = None


def device():
    global DEV
    if DEV is None:
        DEV = Device()
    return DEV


def device_shot(bg, screen, pose, glow=0.55, refl=0.30, floor=True, rim_light=None):
    """Composite the phone (showing `screen`) over `bg`. Returns (frame, homography)."""
    d = device()
    frame = bg.copy()
    tex = d.texture(screen)
    Hm = d.homography(pose)
    warped = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = warped[..., 3:4]
    rgb = warped[..., :3]
    fy = d.floor_y(pose)
    x, y, z = pose[:3]
    horizon = proj(np.array([[x, fy, z]]))[0, 1]
    if floor:
        # glossy dark floor below the contact line
        ys = np.arange(H, dtype=np.float32)
        k = np.clip((ys - (horizon - P(60))) / P(260), 0, 1)
        frame *= (1 - 0.72 * k)[:, None, None]
        # reflection of the device in the floor
        Hr = d.homography(pose, mirror_y=fy)
        wr = cv2.warpPerspective(tex, Hr, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fade = np.clip(1 - (ys - horizon) / P(900), 0, 1) ** 1.6 * (ys > horizon - P(4))
        ra = wr[..., 3:4] * (fade * refl)[:, None, None]
        wr_rgb = cv2.GaussianBlur(wr[..., :3], (0, 0), max(0.5, P(5)))
        frame[:] = frame * (1 - ra) + wr_rgb * ra
    # screen light spilling into the arena (and pooling on the floor)
    glow_add(frame, rgb * a, glow * 0.7, sigmas=(70, 220))
    if floor:
        small = cv2.resize(rgb * a, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
        pool = cv2.GaussianBlur(small, (0, 0), max(0.6, P(140) / 16))
        pool = cv2.resize(pool, (W, H))
        hy = int(max(0, min(H - 1, horizon)))
        lower = np.zeros_like(frame)
        span = H - hy
        if span > 0:
            src = pool[max(0, hy - span):hy][::-1]
            lower[hy:hy + src.shape[0]] = src
        frame += lower * 0.55
    # device thickness (side walls), lit faintly by rim light colour
    hull = cv2.convexHull(np.vstack([proj(d.outline(pose)), proj(d.outline(pose, d.th))]).astype(np.int32))
    side = np.array((0.05, 0.05, 0.055)) if rim_light is None else 0.04 + 0.12 * np.array(rim_light)
    cv2.fillConvexPoly(frame, hull, tuple(float(v) for v in side), cv2.LINE_AA)
    frame[:] = frame * (1 - a) + rgb * a
    if rim_light is not None:
        edge = cv2.morphologyEx(warped[..., 3], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        glow_add(frame, edge[..., None] * np.array(rim_light, np.float32), 1.0, sigmas=(6, 30), down=4)
    return frame, Hm


def screen_glow(img, rect, color, k, width=26, inner=0.0):
    """Additive rim light around a rectangle on the recording (lighting only, layout untouched)."""
    if k <= 0.001:
        return img
    x0, y0, x1, y1 = rect
    sm = np.zeros((SRC_H // 4, SRC_W // 4, 3), np.float32)
    cv2.rectangle(sm, (x0 // 4, y0 // 4), (x1 // 4, y1 // 4), color, max(1, width // 4), cv2.LINE_AA)
    g = cv2.GaussianBlur(sm, (0, 0), 3) * 1.6 + cv2.GaussianBlur(sm, (0, 0), 14) * 2.2
    if inner > 0:
        fill = np.zeros_like(sm)
        cv2.rectangle(fill, (x0 // 4, y0 // 4), (x1 // 4, y1 // 4), color, -1)
        g += cv2.GaussianBlur(fill, (0, 0), 10) * inner
    g = cv2.resize(g, (img.shape[1], img.shape[0]))
    return img + g * k


# ---------------------------------------------------------------- particles
class Embers:
    def __init__(self, seed, n, size=(2, 6), speed=(120, 420), color=(0.25, 0.55, 1.0), down=3,
                 blur=1.2, gain=1.0):
        r = np.random.default_rng(seed)
        self.n = n
        self.x0, self.y0 = r.uniform(0, 2400, n), r.uniform(0, 4200, n)
        self.v = r.uniform(*speed, n)
        self.sz = r.uniform(*size, n)
        self.ph = r.uniform(0, 6.28, n)
        self.fr = r.uniform(0.6, 2.5, n)
        self.sw = r.uniform(10, 70, n)
        self.dr = r.uniform(-50, 50, n)
        self.hue = r.uniform(0, 1, n)
        self.color = np.array(color, np.float32)
        self.down, self.blur, self.gain = down, blur, gain

    def draw(self, frame, t, ox=0.0, oy=0.0, gain=1.0):
        d = self.down
        w, h = W // d, H // d
        y = (self.y0 - self.v * t + oy) % 4200 - 180
        x = (self.x0 + self.dr * t + self.sw * np.sin(self.ph + t * self.fr) + ox) % 2400 - 120
        fl = 0.45 + 0.55 * np.abs(np.sin(self.ph * 3 + t * self.fr * 4))
        layer = np.zeros((h, w, 3), np.float32)
        gold = np.array((0.35, 0.78, 1.0), np.float32)
        for i in range(self.n):
            col = self.color * (1 - self.hue[i] * 0.5) + gold * self.hue[i] * 0.5
            cv2.circle(layer, (int(P(x[i]) / d), int(P(y[i]) / d)), max(1, int(P(self.sz[i]) / d)),
                       tuple(float(c) for c in col * fl[i]), -1, cv2.LINE_AA)
        if self.blur > 0:
            layer = cv2.GaussianBlur(layer, (0, 0), max(0.5, P(self.blur * 3) / d))
        frame += cv2.resize(layer, (W, H)) * self.gain * gain


def splat(frame, xs, ys, cols, sigma_core=1.0, sigma_glow=7.0, glow=0.9, down=2):
    """Additively draw many point particles (vectorised) with a soft glow."""
    w, h = W // down, H // down
    xi = (xs / down).astype(np.int32)
    yi = (ys / down).astype(np.int32)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    layer = np.zeros((h, w, 3), np.float32)
    np.add.at(layer, (yi[ok], xi[ok]), cols[ok])
    core = cv2.GaussianBlur(layer, (0, 0), max(0.5, P(sigma_core * 2) / down))
    g = cv2.GaussianBlur(layer, (0, 0), max(0.8, P(sigma_glow * 2) / down))
    frame += cv2.resize(core * 3.0 + g * glow * 6.0, (W, H))


# ---------------------------------------------------------------- type
def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), max(4, int(size)))


@functools.lru_cache(maxsize=None)
def title_img(lines, size=250, bar=(200, 20, 30), fg=(255, 255, 255), font_name='Anton.ttf', pad_x=70,
              pad_y=34, gap=22, extrude=12, bar_on=True):
    """Bold condensed white type with a short extrusion, each line on a clean red banner bar."""
    f = font(font_name, P(size))
    px, py, g, ex = int(P(pad_x)), int(P(pad_y)), int(P(gap)), max(1, int(P(extrude)))
    boxes = [f.getbbox(t) for t in lines]
    ws = [b[2] - b[0] for b in boxes]
    cap = max(b[3] - b[1] for b in boxes)
    bar_h = cap + 2 * py + ex
    bw = max(ws) + 2 * px
    marg = int(P(80))
    Wc, Hc = bw + 2 * marg, len(lines) * bar_h + (len(lines) - 1) * g + 2 * marg
    shadow = Image.new('L', (Wc, Hc), 0)
    bars = Image.new('RGBA', (Wc, Hc), (0, 0, 0, 0))
    text = Image.new('RGBA', (Wc, Hc), (0, 0, 0, 0))
    ds, db, dt = ImageDraw.Draw(shadow), ImageDraw.Draw(bars), ImageDraw.Draw(text)
    for i, (t, b, w_) in enumerate(zip(lines, boxes, ws)):
        y0 = marg + i * (bar_h + g)
        lw = w_ + 2 * px
        x0 = (Wc - lw) // 2
        if bar_on:
            ds.rectangle([x0 + int(P(10)), y0 + int(P(18)), x0 + lw + int(P(10)), y0 + bar_h + int(P(18))], fill=200)
            for k in range(bar_h):
                sh = 1.0 - 0.28 * (k / bar_h)
                db.line([(x0, y0 + k), (x0 + lw, y0 + k)],
                        fill=(int(bar[0] * sh), int(bar[1] * sh), int(bar[2] * sh), 255))
            db.line([(x0, y0), (x0 + lw, y0)], fill=(255, 120, 110, 255), width=max(1, int(P(4))))
        else:
            ds.text((x0 + px - b[0] + int(P(8)), y0 + py - b[1] + int(P(14))), t, font=f, fill=220)
        tx, ty = x0 + px - b[0], y0 + py - b[1]
        for k in range(ex, 0, -1):
            v = int(30 + 40 * (1 - k / ex))
            dt.text((tx, ty + k), t, font=f, fill=(v, v, v + 4, 255))
        dt.text((tx, ty), t, font=f, fill=fg + (255,))
    from PIL import ImageFilter
    shadow = shadow.filter(ImageFilter.GaussianBlur(P(22)))
    base = Image.new('RGBA', (Wc, Hc), (0, 0, 0, 0))
    base.putalpha(shadow.point(lambda v: int(v * 0.7)))
    base = Image.alpha_composite(base, bars)
    base = Image.alpha_composite(base, text)
    arr = np.asarray(base).astype(np.float32) / 255.0
    rgb = arr[..., [2, 1, 0]]
    # subtle vertical sheen on the type face
    return np.dstack([rgb, arr[..., 3]])


def draw_title(frame, lines, t, t0, cy, t1=None, size=250, cx=None, bar=(200, 20, 30), fg=(255, 255, 255),
               font_name='Anton.ttf', shake=18, bar_on=True):
    tau = t - t0
    if tau < 0 or (t1 is not None and t > t1 + 0.16):
        return
    img = title_img(tuple(lines), size, bar, fg, font_name, bar_on=bar_on)
    cx = W / 2 if cx is None else cx
    s = 1 + 0.85 * (1 - eo3(tau / 0.12))
    a = eo3(tau / 0.07)
    ox = oy = 0.0
    if tau > 0.12:
        k = tau - 0.12
        amp = P(shake) * math.exp(-k / 0.08)
        ox, oy = amp * math.sin(k * 97), amp * 0.6 * math.cos(k * 83)
    if t1 is not None and t > t1:
        e = eo3((t - t1) / 0.16)
        s *= 1 + 0.35 * e
        a *= 1 - e
    place(frame, img, cx + ox, cy + oy, s, a)
    if 0.1 < tau < 0.4:  # brief hot flash on the type as it lands
        place(frame, img, cx + ox, cy + oy, s, 0.6 * math.exp(-(tau - 0.1) / 0.07), add=True)


@functools.lru_cache(maxsize=None)
def text_img(text, size, font_name='Oswald-Bold.ttf', fg=(255, 255, 255), tracking=0):
    f = font(font_name, P(size))
    b = f.getbbox(text)
    w, h = b[2] - b[0] + int(P(20)) + tracking * len(text), b[3] - b[1] + int(P(20))
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if tracking:
        x = int(P(10)) - b[0]
        for ch in text:
            d.text((x, int(P(10)) - b[1]), ch, font=f, fill=fg + (255,))
            x += f.getlength(ch) + tracking
    else:
        d.text((int(P(10)) - b[0], int(P(10)) - b[1]), text, font=f, fill=fg + (255,))
    arr = np.asarray(im).astype(np.float32) / 255.0
    return np.dstack([arr[..., [2, 1, 0]], arr[..., 3]])


# ---------------------------------------------------------------- arena / grade
@functools.lru_cache(maxsize=None)
def arena_small(name):
    img = cv2.imread(os.path.join(SRC, 'stills', name)).astype(np.float32) / 255.0
    small = cover(img, out=(W // 6, H // 6), zoom=1.25)
    return cv2.GaussianBlur(small, (0, 0), max(0.8, P(36) / 6))


def arena(name, t, ox=0.0, oy=0.0, dim=0.30, zoom=1.0):
    sm = arena_small(name)
    h, w = sm.shape[:2]
    s = zoom * (1.12 + 0.01 * t)
    M = np.array([[s, 0, w / 2 - s * w / 2 + P(ox) / 6], [0, s, h / 2 - s * h / 2 + P(oy) / 6]], np.float32)
    sm = cv2.warpAffine(sm, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(sm, (W, H)) * dim


@functools.lru_cache(maxsize=None)
def vignette_map(strength=0.6):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.58)) ** 2)
    return np.clip(1 - strength * np.clip(r - 0.35, 0, None) ** 1.6, 0.18, 1)[..., None]


@functools.lru_cache(maxsize=None)
def grain_tiles(n=6):
    r = np.random.default_rng(7)
    out = []
    for _ in range(n):
        g = r.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
        out.append(cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None])
    return out


def light_leak(frame, t, k=0.10, warm=(0.25, 0.45, 1.0)):
    sw, sh = W // 16, H // 16
    layer = np.zeros((sh, sw, 3), np.float32)
    x = sw * (0.5 + 0.55 * math.sin(t * 0.37 + 1.0))
    y = sh * (0.18 + 0.12 * math.sin(t * 0.23))
    cv2.circle(layer, (int(x), int(y)), int(sw * 0.45), warm, -1)
    x2 = sw * (0.5 + 0.6 * math.cos(t * 0.29 + 2.2))
    cv2.circle(layer, (int(x2), int(sh * 0.9)), int(sw * 0.4), (0.15, 0.25, 0.9), -1)
    layer = cv2.GaussianBlur(layer, (0, 0), sw * 0.18)
    frame += cv2.resize(layer, (W, H)) * k


def grade(f, t, sat=1.18, leak=0.10, vig=0.62, grain=0.018, exposure=1.0, warm=0.0, bloom=0.5):
    f *= exposure
    if warm:
        f *= np.array([1 - 0.12 * warm, 1.0, 1 + 0.10 * warm], np.float32)
    small = cv2.resize(f, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    br = np.maximum(small - 0.6, 0)
    b = cv2.GaussianBlur(br, (0, 0), max(0.6, P(36) / 8)) + 0.6 * cv2.GaussianBlur(br, (0, 0), max(0.8, P(140) / 8))
    f += cv2.resize(b, (W, H)) * bloom
    light_leak(f, t, leak)
    lum = f.mean(2, keepdims=True)
    f = lum + (f - lum) * sat
    np.clip(f, 0, 6, out=f)
    f = f * (1 + f / 4.0) / (1 + f)          # soft highlight roll-off
    f = np.power(f, 1.12)                     # deeper blacks
    # split tone: cool shadows, warm/gold highlights
    lum = f.mean(2, keepdims=True)
    f += (0.025 * (1 - lum) * np.array([1.0, 0.2, -0.4], np.float32) +
          0.03 * lum * np.array([-0.5, 0.15, 0.6], np.float32))
    f *= vignette_map(vig)
    f += grain_tiles()[int(t * FPS) % 6] * grain
    return np.clip(f, 0, 1)


def to_u8(f):
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
