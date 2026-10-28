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

    def texture(self, screen, glint=0.0, env=None):
        TW, TH, BZ = self.TW, self.TH, self.BZ
        scr = cv2.resize(screen, (self.SW, self.SH), interpolation=cv2.INTER_AREA if SC < 1 else cv2.INTER_LINEAR)
        full = np.zeros((TH, TW, 3), np.float32)
        full[BZ:BZ + self.SH, BZ:BZ + self.SW] = scr
        rgb = self.body_rgb * (1 - self.screen_mask) + full * self.screen_mask
        if env is not None:   # the room reflected in the cover glass
            rgb += cv2.resize(env, (TW, TH))[::-1] * 0.07 * self.screen_mask
        if not hasattr(self, '_dd'):
            yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
            self._dd = xx / TW * 0.8 + yy / TH * 0.6
        d = self._dd - glint
        gl = (np.exp(-((d - 0.38) / 0.09) ** 2) * 0.06 + np.exp(-((d - 0.55) / 0.025) ** 2) * 0.05)[..., None]
        rgb = rgb + gl * self.screen_mask
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
    tex = d.texture(screen, glint=pose[3] * 0.012, env=cv2.resize(bg, (W // 8, H // 8)) if isinstance(d, Device) and not isinstance(d, Monitor) else None)
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


GOLD_RGB = (206, 172, 98)
POST = []  # text layers composited after the grade so white stays clean white


def place_post(img, cx, cy, s=1.0, alpha=1.0):
    POST.append((img, cx, cy, s, alpha))


def flush_post(frame):
    for item in POST:
        if callable(item):
            item(frame)
        else:
            img, cx, cy, s, a = item
            place(frame, img, cx, cy, s, a)
    POST.clear()
    return frame


HEAD_FONT = 'Anton.ttf'   # the bold condensed caption type of the V1/V2 edits


@functools.lru_cache(maxsize=None)
def text_mask(lines, size, font_name=HEAD_FONT, tracking=0.03, gap=0.14):
    """Anti-aliased mask of centred multi-line type with letter-spacing (in em)."""
    f = font(font_name, P(size))
    tr = P(size) * tracking
    ws = [sum(f.getlength(ch) for ch in t) + tr * (len(t) - 1) for t in lines]
    asc, desc = f.getmetrics()
    cap = f.getbbox('H')[3] - f.getbbox('H')[1]
    line_h = cap
    g = int(P(size) * gap)
    marg = int(P(60))
    Wc = int(max(ws)) + 2 * marg
    Hc = len(lines) * line_h + (len(lines) - 1) * g + 2 * marg
    im = Image.new('L', (Wc, Hc), 0)
    d = ImageDraw.Draw(im)
    top_off = f.getbbox('H')[1]
    for i, (t, w_) in enumerate(zip(lines, ws)):
        x = (Wc - w_) / 2
        y = marg + i * (line_h + g) - top_off
        for ch in t:
            d.text((x, y), ch, font=f, fill=255)
            x += f.getlength(ch) + tr
    return np.asarray(im).astype(np.float32) / 255.0, max(ws)


@functools.lru_cache(maxsize=None)
def title_img(lines, size=200, font_name=HEAD_FONT, accent=True, tracking=0.03, color=(0.97, 0.96, 0.94)):
    """Premium caption: classical serif caps, refined tracking, one thin gold rule, soft dark shadow."""
    tm, tw = text_mask(tuple(lines), size, font_name, tracking)
    h, w = tm.shape
    acc_gap, acc_h = int(P(size * 0.22)), max(1, int(P(5)))
    pad_b = acc_gap + acc_h + int(P(10)) if accent else 0
    tm = np.pad(tm, ((0, pad_b), (0, 0)))
    am = np.zeros_like(tm)
    if accent:
        ya = h - int(P(60)) + acc_gap
        acc_w = int(min(tw * 0.5, P(420)))
        am[ya:ya + acc_h, (w - acc_w) // 2:(w + acc_w) // 2] = 1.0
        # rule tapers at its ends
        xs = np.arange(w, dtype=np.float32)
        taper = np.clip(1 - np.abs(xs - w / 2) / (acc_w / 2 + 1e-3), 0, 1) ** 0.35
        am *= taper[None, :]
    sil = np.maximum(tm, am)
    sh = np.roll(sil, int(P(6)), 0)
    shadow = np.clip(cv2.GaussianBlur(sh, (0, 0), P(6)) * 0.7 + cv2.GaussianBlur(sh, (0, 0), P(28)) * 0.55, 0, 0.8)
    gold = np.array(GOLD_RGB[::-1], np.float32) / 255.0
    white = np.array(color[::-1], np.float32)
    # very subtle top-to-bottom tonal falloff on the face (no gradient fill, just material)
    ys = np.linspace(1.0, 0.93, tm.shape[0], dtype=np.float32)[:, None, None]
    rgb = np.where(sil[..., None] > 0, (tm[..., None] * white * ys + am[..., None] * gold) / np.maximum(sil[..., None], 1e-4), 0)
    alpha = np.clip(sil + shadow * (1 - sil), 0, 1)
    return np.dstack([rgb.astype(np.float32), alpha.astype(np.float32)]), am


def _title_anim(t, t0, t1):
    tau = t - t0
    e = 1 - math.exp(-tau / 0.055)          # heavy exponential settle
    s = 1 + 0.32 * (1 - e)
    a = clamp01(tau / 0.08)
    blur = (1 - e) * 1.0
    oy = 0.0
    if t1 is not None and t > t1:
        k = eio((t - t1) / 0.2)
        a *= 1 - k
        blur = max(blur, k * 0.8)
        oy = -P(30) * k
        s *= 1 + 0.04 * k
    else:
        s *= 1 - 0.012 * clamp01(tau / 2.0)    # slow weighted drift while holding
    return tau, s, a, blur, oy


def draw_title(frame, lines, t, t0, cy, t1=None, size=200, cx=None, shake=10, accent=True, glass=False, **_):
    """Slam in with motion blur and a short impact shake, hold with a slow weighted drift, ease out."""
    tau = t - t0
    if tau < 0 or (t1 is not None and t > t1 + 0.2):
        return
    cx = W / 2 if cx is None else cx
    tau, s, a, blur, oy = _title_anim(t, t0, t1)
    ox = 0.0
    if 0.06 < tau < 0.35:
        k = tau - 0.06
        amp = P(shake) * math.exp(-k / 0.05)
        ox, oy = ox + amp * math.sin(k * 90), oy + amp * 0.5 * math.cos(k * 77)
    if glass:
        POST.append(lambda fr, L=tuple(lines), z=size, x=cx + ox, y=cy + oy, sc=s, al=a, bl=blur, tu=tau:
                    glass_text(fr, L, z, x, y, sc, al, bl, tu))
        return
    img, am = title_img(tuple(lines), size, accent=accent)
    if accent:  # the gold rule draws outward after the type lands
        k = eo3((tau - 0.12) / 0.3)
        img = img.copy()
        h, w = am.shape
        xs = np.abs(np.arange(w) - w / 2)
        keep = (xs <= k * w / 2).astype(np.float32)
        rule = am > 0
        img[..., 3] = np.where(rule, img[..., 3] * keep[None, :], img[..., 3])
    if blur > 0.03:
        img = img.copy()
        k = max(1, int(P(40) * blur))
        img = cv2.blur(img, (1, k * 2 + 1))
    POST.append((img, cx + ox, cy + oy, s, a))
    q = (tau - 0.14) / 0.45
    if 0 < q < 1.15:   # light catches the type once, left to right
        h, w = img.shape[:2]
        d = np.arange(w, dtype=np.float32)[None, :] + np.arange(h, dtype=np.float32)[:, None] * 0.5
        span = w + h * 0.5
        band_ = np.exp(-((d - (q * 1.3 - 0.15) * span) / (span * 0.035)) ** 2)
        shine = np.dstack([np.ones((h, w, 3), np.float32), band_ * np.clip(am * 0 + img[..., 3] * (img[..., :3].mean(2) > 0.5), 0, 1) * 0.55])
        POST.append((shine, cx + ox, cy + oy, s, a))


def glass_text(frame, lines, size, cx, cy, s, a, blur, tau=1.0):
    """Frosted glass letters: refract and frost what is behind them, bevel highlight, crisp rim."""
    m, _ = text_mask(tuple(lines), size)
    if s != 1.0:
        m = cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    if blur > 0.03:
        k = max(1, int(P(40) * blur))
        m = cv2.blur(m, (1, k * 2 + 1))
    h, w = m.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    m = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * a
    roi = frame[Y0:Y1, X0:X1]
    mb = cv2.GaussianBlur(m, (0, 0), P(7))
    gx = cv2.Sobel(mb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mb, cv2.CV_32F, 0, 1, ksize=3)
    hh, ww = m.shape
    mx, my = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
    frost = cv2.GaussianBlur(roi, (0, 0), P(16))
    refr = cv2.remap(frost, mx + gx * P(60), my + gy * P(60), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    glass = refr * 1.45 + np.array([0.16, 0.16, 0.17], np.float32)
    light = np.clip(-(gx + gy) * P(14), -1, 1)[..., None]       # light from the top-left
    edge = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(P(2)))), 0, 1)[..., None]
    sh = cv2.GaussianBlur(np.roll(m, int(P(10)), 0), (0, 0), P(18))[..., None] * 0.55
    roi[:] = roi * (1 - sh * (1 - m[..., None]))
    mm = m[..., None]
    roi[:] = roi * (1 - mm) + glass * mm
    roi += (np.clip(light, 0, 1) * 0.45 - np.clip(-light, 0, 1) * 0.22) * mm + edge * 0.8
    # a specular band slides across the glass as it lands, then a slow faint shimmer
    q = (tau - 0.08) / 0.55
    xs = np.arange(ww, dtype=np.float32)[None, :]
    ys_ = np.arange(hh, dtype=np.float32)[:, None]
    d = xs + ys_ * 0.6
    span = ww + hh * 0.6
    sweep = np.exp(-((d - (q * 1.3 - 0.15) * span) / (span * 0.05)) ** 2) * (0 < q < 1.2)
    shimmer = 0.06 * (0.5 + 0.5 * np.sin(d / span * 6.28 - tau * 1.8))
    roi += ((sweep * 0.75 + shimmer) * m)[..., None]


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


# ---------------------------------------------------------------- gaming PC setup
class Monitor(Device):
    """Thin-bezel desktop monitor; same screen-mapping interface as the phone."""

    def __init__(self):
        self.SW, self.SH = int(P(2100)), int(P(2100) * SRC_H / SRC_W)
        self.BZ = int(P(26))
        chin = int(P(46))
        self.TW, self.TH = self.SW + 2 * self.BZ, self.SH + self.BZ + chin
        TW, TH, BZ = self.TW, self.TH, self.BZ
        body = rrect_mask(TW, TH, int(P(18)))
        rgb = np.zeros((TH, TW, 3), np.float32)
        rgb[:] = (0.035, 0.035, 0.04)
        edge = np.clip(body - cv2.erode(body, np.ones((3, 3), np.uint8), iterations=max(1, int(P(3)))), 0, 1)
        rgb += edge[..., None] * 0.22
        rgb[TH - chin + int(P(6)):TH - int(P(6)), :] *= 0.8
        self.body_rgb, self.body_a = rgb, body
        sm = np.zeros((TH, TW), np.float32)
        sm[BZ:BZ + self.SH, BZ:BZ + self.SW] = 1.0
        self.screen_mask = sm[..., None]
        yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
        d = xx / TW * 0.7 + yy / TH * 0.5
        self.gloss = (np.exp(-((d - 0.42) / 0.12) ** 2) * 0.035)[..., None] * self.screen_mask
        self.dw = 2.0
        self.dh = self.dw * TH / TW
        self.th = 0.05
        self.r_world = 0.02

    def outline(self, pose, z_off=0.0):
        x, y, z, yaw, pitch, roll = pose
        hx, hy = self.dw / 2, self.dh / 2
        loc = np.array([[-hx, -hy, z_off], [hx, -hy, z_off], [hx, hy, z_off], [-hx, hy, z_off]])
        return (rotm(yaw, pitch, roll) @ loc.T).T + np.array([x, y, z])


MON = None


def monitor():
    global MON
    if MON is None:
        MON = Monitor()
    return MON


@functools.lru_cache(maxsize=None)
def kb_base():
    KW, KH = int(P(1500)), int(P(470))
    body = rrect_mask(KW, KH, int(P(26)))
    keys = np.zeros((KH, KW), np.float32)
    pad, gapk = P(28), P(9)
    rows = [15, 15, 14, 13, 12, 8]
    rh = (KH - 2 * pad - 5 * gapk) / 6
    for r, n in enumerate(rows):
        y0 = pad + r * (rh + gapk)
        if r == 5:
            widths = [1, 1, 1, 6.5, 1, 1, 1, 1]
        else:
            widths = [1] * n
            widths[-1] += (15 - sum(widths))
        unit = (KW - 2 * pad - (len(widths) - 1) * gapk) / sum(widths)
        x0 = pad
        for wk in widths:
            x1 = x0 + wk * unit
            cv2.rectangle(keys, (int(x0), int(y0)), (int(x1), int(y0 + rh)), 1.0, -1)
            x0 = x1 + gapk
    return KW, KH, body, keys


def kb_texture(t):
    KW, KH, body, keys = kb_base()
    hue = ((np.arange(KW, dtype=np.float32) / KW * 180 + t * 50) % 180).astype(np.uint8)
    hsv = np.stack([hue, np.full(KW, 255, np.uint8), np.full(KW, 255, np.uint8)], 1)[None]
    row = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0].astype(np.float32) / 255.0
    light = np.repeat(row[None], KH, 0)
    under = cv2.GaussianBlur(keys, (0, 0), max(1.0, P(5))) - keys * 0.85
    rgb = np.full((KH, KW, 3), 0.03, np.float32)
    rgb += np.clip(under, 0, 1)[..., None] * light * 1.3
    rgb = rgb * (1 - keys[..., None]) + (0.07 + light * 0.10) * keys[..., None]
    return np.dstack([rgb, body]), light * keys[..., None]


def _poly_proj(R, T, loc):
    Pw = (R @ np.asarray(loc, float).T).T + T
    return Pw, proj(Pw)


def desk_shot(bg, screen, pose, t, light=(0.95, 0.25, 0.7), glow=0.5, refl=0.22, kb=True):
    """Gaming PC setup: monitor on a stand, RGB keyboard and mouse on a glossy desk, LED wall glow."""
    m = monitor()
    frame = bg.copy()
    x, y, z, yaw, pitch, roll = pose
    R, T = rotm(yaw, pitch, roll), np.array([x, y, z])
    light = np.array(light, np.float32)
    desk_y = m.dh / 2 + 0.42
    # LED strip behind the monitor washing the wall
    _, c = _poly_proj(R, T, [[0, -0.1, 0.55]])
    lay = np.zeros((H // 8, W // 8, 3), np.float32)
    cv2.ellipse(lay, (int(c[0, 0] / 8), int(c[0, 1] / 8)), (int(W * 0.62 / 8), int(H * 0.17 / 8)), 0, 0, 360,
                tuple(float(v) for v in light), -1)
    lay = cv2.GaussianBlur(lay, (0, 0), W / 8 * 0.16)
    frame += cv2.resize(lay, (W, H)) * 0.75
    # desk surface below its back edge
    _, e = _poly_proj(R, T, [[-6, desk_y, 0.7], [6, desk_y, 0.7]])
    (x1, y1), (x2, y2) = e
    yl = y1 + (0 - x1) * (y2 - y1) / (x2 - x1 + 1e-6)
    yr = y1 + (W - x1) * (y2 - y1) / (x2 - x1 + 1e-6)
    dm = np.zeros((H, W), np.float32)
    cv2.fillPoly(dm, [np.array([[0, yl], [W, yr], [W, H + 10], [0, H + 10]], np.int32)], 1.0, cv2.LINE_AA)
    ys = np.arange(H, dtype=np.float32)[:, None, None]
    yedge = (yl + yr) / 2
    deskcol = 0.022 + light * 0.20 * np.exp(-np.clip(ys - yedge, 0, None) / P(420))
    frame = frame * (1 - dm[..., None]) + deskcol * dm[..., None]
    # monitor reflection in the glossy desk
    tex = m.texture(screen)
    hx, hy = m.dw / 2, m.dh / 2
    mir = [[-hx, 2 * desk_y + hy, 0], [hx, 2 * desk_y + hy, 0], [hx, 2 * desk_y - hy, 0], [-hx, 2 * desk_y - hy, 0]]
    Pm, pm = _poly_proj(R, T, mir)
    if (Pm[:, 2] > 0.3).all():
        src = np.array([[0, 0], [m.TW, 0], [m.TW, m.TH], [0, m.TH]], np.float32)
        Hr = cv2.getPerspectiveTransform(src, pm.astype(np.float32))
        wr = cv2.warpPerspective(tex, Hr, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fade = np.clip(1 - (ys[..., 0] - yedge) / P(700), 0, 1) ** 1.5
        ra = wr[..., 3:4] * fade[..., None] * refl * dm[..., None]
        frame[:] = frame * (1 - ra) + cv2.GaussianBlur(wr[..., :3], (0, 0), max(0.5, P(6))) * ra
    # stand: base plate and neck
    Pb, pb = _poly_proj(R, T, [[-0.36, desk_y, -0.02], [0.36, desk_y, -0.02], [0.3, desk_y, 0.32], [-0.3, desk_y, 0.32]])
    if (Pb[:, 2] > 0.3).all():
        cv2.fillConvexPoly(frame, pb.astype(np.int32), (0.05, 0.05, 0.055), cv2.LINE_AA)
        cv2.line(frame, tuple(pb[0].astype(int)), tuple(pb[1].astype(int)), (0.25, 0.25, 0.27), max(1, int(P(3))), cv2.LINE_AA)
    Pn, pn = _poly_proj(R, T, [[-0.07, hy - 0.2, 0.07], [0.07, hy - 0.2, 0.07], [0.09, desk_y, 0.12], [-0.09, desk_y, 0.12]])
    if (Pn[:, 2] > 0.3).all():
        cv2.fillConvexPoly(frame, pn.astype(np.int32), (0.07, 0.07, 0.078), cv2.LINE_AA)
    # monitor body + screen
    hull = cv2.convexHull(np.vstack([proj(m.outline(pose)), proj(m.outline(pose, m.th))]).astype(np.int32))
    cv2.fillConvexPoly(frame, hull, tuple(float(v) for v in (0.03 + 0.08 * light)), cv2.LINE_AA)
    Hm = m.homography(pose)
    warped = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = warped[..., 3:4]
    glow_add(frame, warped[..., :3] * a, glow * 0.7, sigmas=(60, 200))
    frame[:] = frame * (1 - a) + warped[..., :3] * a
    # keyboard + mouse on the desk
    if kb:
        ky = desk_y - 0.03
        Pk, pk = _poly_proj(R, T, [[-0.86, ky, -0.6], [0.86, ky, -0.6], [0.86, ky, -1.1], [-0.86, ky, -1.1]])
        if (Pk[:, 2] > 0.35).all():
            ktex, klight = kb_texture(t)
            src = np.array([[0, 0], [ktex.shape[1], 0], [ktex.shape[1], ktex.shape[0]], [0, ktex.shape[0]]], np.float32)
            Hk = cv2.getPerspectiveTransform(src, pk.astype(np.float32))
            _, pf = _poly_proj(R, T, [[0.86, desk_y, -1.1], [-0.86, desk_y, -1.1]])
            cv2.fillConvexPoly(frame, np.array([pk[2], pk[3], pf[1], pf[0]], np.int32), (0.02, 0.02, 0.025), cv2.LINE_AA)
            kw = cv2.warpPerspective(ktex, Hk, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            kl = cv2.warpPerspective(np.ascontiguousarray(klight), Hk, (W, H), flags=cv2.INTER_LINEAR)
            glow_add(frame, kl, 0.9, sigmas=(25, 90), down=4)
            ka = kw[..., 3:4]
            frame[:] = frame * (1 - ka) + kw[..., :3] * ka
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        Pmo, pmo = _poly_proj(R, T, np.stack([1.2 + 0.1 * np.cos(ang), np.full_like(ang, desk_y - 0.035),
                                               -0.85 + 0.16 * np.sin(ang)], 1))
        if (Pmo[:, 2] > 0.35).all():
            cv2.fillConvexPoly(frame, cv2.convexHull(pmo.astype(np.int32)), (0.035, 0.035, 0.04), cv2.LINE_AA)
            cv2.polylines(frame, [cv2.convexHull(pmo.astype(np.int32))], True,
                          tuple(float(v) for v in light * 0.9), max(1, int(P(4))), cv2.LINE_AA)
    return frame, Hm
