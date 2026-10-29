"""Locked type system for "The Perfect Move" (see TYPE_SPEC.md).

One fixed narrator size, one fixed thought size, one text zone (top), everything centre aligned.
Units are final 4K units (2160x3840) like trailer/common.py; P() scales them for previews.
"""
import functools
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trailer'))
import common as C  # noqa: E402
from common import P  # noqa: E402

NARR_FONT = 'Cinzel_wght_900.ttf'
NARR_SIZE = 172            # the one narrator size (86 px font at 1080): largest size where every line clears the right UI zone
NARR_TRACK = 0.06
THOUGHT_FONT = 'CormorantGaramond-BoldItalic.ttf'
THOUGHT_SIZE = 172         # the one thought size (86 px font at 1080)
TEXT_TOP = 520             # the one text zone: the first line's cap top always sits 260 px from the top at 1080x1920
RULE_W, RULE_H, RULE_GAP = 380, 7, 40   # gold rule: fixed width, weight and spacing under every narrator block
GOLD = np.array(C.GOLD_RGB[::-1], np.float32) / 255.0
IVORY = np.array((0.94, 0.96, 0.97), np.float32)       # BGR ivory
SAFE_W = 1600              # 800 px usable width at 1080 (frame centre, right UI zone 140 px)


def _mask(lines, font_name, size, tracking, gap, italic_pad=0.0):
    f = C.font(font_name, P(size))
    tr = P(size) * tracking
    ws = [sum(f.getlength(ch) for ch in t) + tr * (len(t) - 1) for t in lines]
    top = f.getbbox('H')[1]
    cap = f.getbbox('H')[3] - top
    asc_desc = f.getbbox('Hgy')[3] - top
    line_h = cap if font_name == NARR_FONT else asc_desc
    g = int(P(size) * gap)
    marg = int(P(60) + P(size) * italic_pad)
    Wc = int(max(ws)) + 2 * marg
    Hc = len(lines) * line_h + (len(lines) - 1) * g + 2 * marg
    im = Image.new('L', (Wc, Hc), 0)
    d = ImageDraw.Draw(im)
    for i, (t, w_) in enumerate(zip(lines, ws)):
        x = (Wc - w_) / 2
        y = marg + i * (line_h + g) - top
        for ch in t:
            d.text((x, y), ch, font=f, fill=255)
            x += f.getlength(ch) + tr
    return np.asarray(im).astype(np.float32) / 255.0, max(ws), marg


@functools.lru_cache(maxsize=None)
def narrator_img(lines):
    """Ivory Cinzel Black caps + centred gold rule (fixed width/weight/gap). Returns (rgba, rule_mask)."""
    tm, tw, marg = _mask(tuple(lines), NARR_FONT, NARR_SIZE, NARR_TRACK, 0.34)
    assert tw <= P(SAFE_W) + 1, (lines, tw)
    h, w = tm.shape
    rw = int(P(RULE_W))
    w2 = max(w, rw + 2 * int(P(60)))
    pad_x = (w2 - w) // 2
    rule_y = h - marg + int(P(RULE_GAP))
    pad_b = int(P(RULE_GAP)) + max(1, int(P(RULE_H))) + int(P(40))
    tm = np.pad(tm, ((0, pad_b), (pad_x, w2 - w - pad_x)))
    am = np.zeros_like(tm)
    xs = np.arange(w2, dtype=np.float32)
    taper = np.clip(1 - np.abs(xs - w2 / 2) / (rw / 2), 0, 1) ** 0.35
    am[rule_y:rule_y + max(1, int(P(RULE_H)))] = taper[None, :]
    return _compose(tm, am, 0.9), am


@functools.lru_cache(maxsize=None)
def thought_img(lines):
    """Ivory bold italic serif, sentence case, soft dark backing for busy boards."""
    tm, tw, _ = _mask(tuple(lines), THOUGHT_FONT, THOUGHT_SIZE, 0.0, 0.0, italic_pad=0.6)
    assert tw <= P(SAFE_W) + 1, (lines, tw)
    back = np.zeros_like(tm)
    k = int(P(60))
    ys, xs = np.nonzero(tm > 0.1)
    cv2.rectangle(back, (int(xs.min()) - k, int(ys.min()) - k // 2), (int(xs.max()) + k, int(ys.max()) + k // 2), 1, -1)
    back = cv2.GaussianBlur(back, (0, 0), P(40)) * 0.58
    return _compose(tm, np.zeros_like(tm), 0.75, extra_alpha=back), None


def _compose(tm, am, shadow_k, extra_alpha=None):
    sil = np.maximum(tm, am)
    sh = np.roll(sil, int(P(6)), 0)
    shadow = np.clip(cv2.GaussianBlur(sh, (0, 0), P(6)) * 0.7 + cv2.GaussianBlur(sh, (0, 0), P(26)) * 0.5, 0, shadow_k)
    if extra_alpha is not None:
        shadow = np.maximum(shadow, extra_alpha)
    rgb = np.where(sil[..., None] > 0,
                   (tm[..., None] * IVORY + am[..., None] * GOLD) / np.maximum(sil[..., None], 1e-4), 0)
    alpha = np.clip(sil + shadow * (1 - sil), 0, 1)
    return np.dstack([rgb.astype(np.float32), alpha.astype(np.float32)])


_TOPS = {}


def place_text(frame, img, top=TEXT_TOP, s=1.0, a=1.0, dx=0.0, dy=0.0):
    """Top-anchored: every block's first cap line starts at the same y (one fixed text place)."""
    k = id(img)
    if k not in _TOPS:
        ink = (img[..., :3].mean(2) > 0.5) & (img[..., 3] > 0.5)
        _TOPS[k] = int(np.nonzero(ink.any(1))[0][0])
    h = img.shape[0]
    cy = P(top) - (_TOPS[k] - h / 2) * s
    C.place(frame, img, C.W / 2 + dx, cy + dy, s, a)


def army_sweep(frame, bgr, k, cy=TEXT_TOP + 150):
    """Army-colour light band sweeping behind the text zone (army colour never touches the letters)."""
    h, w = frame.shape[:2]
    yy = (np.arange(h, dtype=np.float32) - P(cy)) / P(420)
    xx = np.arange(w, dtype=np.float32) / w
    band = np.exp(-yy[:, None] ** 2) * np.exp(-((xx[None, :] - (k * 1.6 - 0.3)) / 0.22) ** 2)
    frame += band[..., None] * (np.array(bgr, np.float32) / 255.0) * 0.55
