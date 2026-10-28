"""Real filmed gaming-PC shots (Mixkit green-screen clips) with the Quess recordings keyed onto the monitor."""
import functools
import json
import math
import os

import cv2
import numpy as np

from common import H, P, SC, SRC, W, affine_cover, band, clip

# ultrawide monitor in Mixkit #51602 / #51614 (screen ~21:9); the 1920x1006 game window is pillarboxed in it
SCREEN_ASPECT = {'pc51602': 2.37, 'pc51614': 1.78}


@functools.lru_cache(maxsize=None)
def track(name):
    return np.array(json.load(open(os.path.join(SRC, name, 'track.json'))), np.float32)


def key_alpha(img):
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    d = g - np.maximum(r, b)
    return np.clip((d - 0.04) / 0.10, 0, 1)


def despill(img, a_screen):
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    lim = np.maximum(r, b)
    g2 = np.where(g > lim, lim + (g - lim) * 0.35, g)
    out = img.copy()
    out[..., 1] = g2
    return out


def plate(name, t):
    """Stock frame + tracked screen quad (in that frame's pixels)."""
    c = clip(name)
    idx = int(min(max(round(t * c.fps), 0), c.n - 1))
    img = c.frame(idx)
    q = track(name)[idx] * (img.shape[1] / 3840.0)
    return img, q


def content_canvas(ui, name, w):
    """Game window letterboxed into the monitor's panel (never cropped or stretched)."""
    asp = SCREEN_ASPECT[name]
    h = int(w / asp)
    canvas = np.full((h, w, 3), 0.012, np.float32)
    uh, uw = ui.shape[:2]
    s = min(w / uw, h / uh)
    r = cv2.resize(ui, (int(uw * s), int(uh * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas, (x0, y0, s)


def screen_on_plate(name, t, ui):
    """Returns the plate with the recording on its monitor, plus a mapper ui-pixel -> plate-pixel."""
    img, q = plate(name, t)
    ph, pw = img.shape[:2]
    cw = int(pw * 0.75)
    canvas, (x0, y0, s) = content_canvas(ui, name, cw)
    ch = canvas.shape[0]
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), q.astype(np.float32))
    warped = cv2.warpPerspective(canvas, Hm, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0.012, 0.012, 0.012))
    a = key_alpha(img)
    a = cv2.GaussianBlur(a, (0, 0), max(0.6, 1.2 * pw / 3840))[..., None]
    base = despill(img, a)
    # monitor light: the screen content now lights the room instead of the green
    sm = cv2.resize(warped * a, (pw // 16, ph // 16), interpolation=cv2.INTER_AREA)
    light = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 6) * 1.6 + cv2.GaussianBlur(sm, (0, 0), 18) * 1.4, (pw, ph))
    base = base + light * 0.22 * (1 - a)
    out = base * (1 - a) + warped * 1.12 * a
    ui_scale = ui.shape[1] / 1920.0   # callers give points in full-res recording pixels

    def ui_to_plate(pts):
        p = np.asarray(pts, np.float32).reshape(-1, 2) * ui_scale
        p = np.stack([x0 + p[:, 0] * s, y0 + p[:, 1] * s], 1)
        return cv2.perspectiveTransform(p[None], Hm)[0]
    return out, ui_to_plate


def pc_frame(name, t, ui, focus_ui=None, zoom=1.0, angle=0.0, wide=False, focus=None, dx=0.0, dy=0.0):
    """Vertical frame of the real PC setup. focus_ui centres the camera on a point of the game screen."""
    out, m = screen_on_plate(name, t, ui)
    ph, pw = out.shape[:2]
    if focus is None:
        fx, fy = m([focus_ui])[0] if focus_ui is not None else (pw / 2, ph / 2)
        focus = (fx / pw, fy / ph)
    if wide:
        return band(out, focus=focus, zoom=zoom, angle=angle, dx=dx, dy=dy, bg_dim=0.35), m
    M = affine_cover(pw, ph, focus, zoom, angle, (W, H), dx, dy)

    def ui_to_out(pts):   # recording pixel -> output frame pixel
        p = m(pts)
        return p @ M[:, :2].T + M[:, 2]
    return cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), ui_to_out
