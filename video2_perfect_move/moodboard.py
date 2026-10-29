"""Moodboard / style frames for "The Perfect Move".

Real frames from the Drive recordings, graded with the trailer grade, captioned with the proposed
type system (narrator: Cinzel caps + gold rule; inner thought: Cormorant Garamond bold italic,
sentence case, soft dark backing). Layout numbers are in 4K units like trailer/common.py.

  SC=0.25 python3 moodboard.py <frames_dir> <out.jpg>
"""
import os
import sys

os.environ.setdefault('SC', '0.25')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trailer'))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import common as C  # noqa: E402
import type2 as T  # noqa: E402
from common import P, W, H  # noqa: E402

FR = sys.argv[1]
SAFE_TOP, SAFE_BOTTOM, SAFE_RIGHT = 440, 760, 280      # 220 / 380 / 140 px at 1080x1920, in 4K units
CX = (2160 - SAFE_RIGHT) / 2 + 60                       # optical centre of the usable area
ARMY = {'FIRE RED': (40, 40, 225), 'WATER BLUE': (235, 150, 30), 'EARTH GREEN': (60, 200, 70),
        'AIR GREY': (200, 200, 200)}


def load(name, focus=(0.5, 0.5), zoom=1.0):
    img = cv2.imread(os.path.join(FR, name + '.jpg')).astype(np.float32) / 255.0
    f = C.cover(img, focus=focus, zoom=zoom)
    f = np.clip((f - 0.5) * 1.08 + 0.5 + 0.03, 0, 1)            # light contrast + shadow lift
    blur = cv2.GaussianBlur(f, (0, 0), 1.2)
    return np.clip(f + (f - blur) * 0.8, 0, 1)                  # light sharpening


def spotlight(f, cx, cy, r, k=0.55):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - P(cx)) ** 2 + (yy - P(cy)) ** 2) / P(r)
    return f * (1 - k * np.clip(d - 1, 0, 1)[..., None]) + 0.12 * np.exp(-d ** 2)[..., None]


def edge_flash(f, bgr, k=0.6):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    e = np.minimum.reduce([xx, yy, W - xx, H - yy]) / P(260)
    return f + np.exp(-e ** 2)[..., None] * np.array(bgr, np.float32) / 255 * k


EMB = C.Embers(11, 70, size=(2, 5), speed=(60, 160), color=(0.35, 0.6, 1.0), down=3, blur=1.6, gain=0.55)


def frame(name, focus, zoom, t, extra=None, narr=None, th=None, flash=None, safe=False):
    f = load(name, focus, zoom)
    if extra:
        f = extra(f)
    if flash:
        f = edge_flash(f, ARMY[flash], 0.45)
        T.army_sweep(f, ARMY[flash], 0.55)
    EMB.draw(f, t)                                   # ember / bokeh depth layer behind the type
    f = C.grade(f, t, sat=1.15, leak=0.06, vig=0.62, grain=0.012, bloom=0.35)
    if narr:
        T.place_text(f, T.narrator_img(narr)[0])
    if th:
        T.place_text(f, T.thought_img(th)[0])
    out = C.to_u8(f)
    if safe:   # UI safe-zone guides (moodboard only)
        o = out.copy()
        cv2.rectangle(o, (0, 0), (W, int(H * 220 / 1920)), (0, 0, 255), -1)
        cv2.rectangle(o, (0, H - int(H * 380 / 1920)), (W, H), (0, 0, 255), -1)
        cv2.rectangle(o, (W - int(W * 140 / 1080), 0), (W, H), (0, 0, 255), -1)
        out = cv2.addWeighted(out, 0.75, o, 0.25, 0)
    return out


TILES = [
    ('1 HOOK  push-in (UI zones in red)', frame('hook', (0.47, 0.55), 1.15, 1.0,
                                                        narr=('WHEN YOU', 'FINALLY FIND'), safe=True)),
    ('2 THOUGHT  FIRE RED turn: sweep + spotlight', frame('red_pov', (0.55, 0.6), 1.35, 3.0,
                                                         extra=lambda f: spotlight(f, 1080, 1700, 520),
                                                         flash='FIRE RED', th=('I know exactly', 'what to do.'))),
    ('3 MOVE PLAYED  overhead hold', frame('board_overhead', (0.5, 0.52), 1.0, 5.0, narr=('MOVE PLAYED.',))),
    ('4 PLAYER 2  WATER BLUE sweep, wide drift', frame('low_blue', (0.42, 0.62), 1.2, 7.0, flash='WATER BLUE',
                                                       narr=('THEN PLAYER 2', 'MOVES.'))),
    ('7 IMPACT  sharp, no blur', frame('fire_char', (0.47, 0.5), 1.25, 9.0, flash='EARTH GREEN',
                                      th=('Wait.', 'That changes everything.'))),
    ('8 CHAOS  native 9:16 hero', frame('chaos1', (0.5, 0.56), 1.0, 11.0, narr=('YOUR PERFECT', 'MOVE...'))),
    ('8 CHAOS  board-level dolly', frame('chaos2', (0.55, 0.55), 1.0, 12.0, narr=('JUST GOT', 'COMPLICATED.'))),
    ('8 PULL-OUT  all four armies', frame('chaos3', (0.5, 0.55), 1.0, 13.0)),
    ('END card  3-line max', frame('thrones', (0.5, 0.55), 1.0, 14.0, narr=('MORE THAN', 'ONE MOVE', 'TO WATCH.'))),
    ('END  real outro + lines', frame('outro', (0.5, 0.62), 1.0, 15.0, narr=('PLAY QUESS.',))),
]

if __name__ == '__main__':
    out = sys.argv[2]
    cols, tw, th_ = 5, W, H
    k = C.SC / 0.25
    lab = int(44 * k)
    F = ImageFont.truetype(os.path.join(C.FONTS, 'Montserrat-SemiBold.ttf'), int(22 * C.SC / 0.25))
    FT = ImageFont.truetype(os.path.join(C.FONTS, 'Cinzel_wght_900.ttf'), int(40 * C.SC / 0.25))
    rows = (len(TILES) + cols - 1) // cols
    ref = [Image.open(os.path.join(FR, f'ref_{i}.jpg')) for i in (1, 10, 22, 27, 30)]
    rw, rh = ref[0].size
    head = int(90 * k)
    ref_h = rh + lab + 20
    sw_h = int(150 * k)
    Wc = cols * tw + (cols + 1) * 16
    Hc = head + ref_h + rows * (th_ + lab + 16) + sw_h + 30
    can = Image.new('RGB', (Wc, Hc), (14, 12, 12))
    d = ImageDraw.Draw(can)
    d.text((16, 22), 'QUESS  |  THE PERFECT MOVE  |  MOODBOARD', font=FT, fill=(206, 172, 98))
    y = head
    d.text((16, y), 'REFERENCE  Quess_Trailer_FINAL_Mobile (match this bar): low macro board, red/teal split grade, '
                    'Cinzel caps + gold rule, deep vignette, bokeh, logo on black', font=F, fill=(220, 220, 220))
    for i, r in enumerate(ref):
        can.paste(r, (16 + i * (rw + 16), y + lab))
    y += ref_h
    for i, (name, img) in enumerate(TILES):
        r, c = divmod(i, cols)
        x0, y0 = 16 + c * (tw + 16), y + r * (th_ + lab + 16)
        d.text((x0, y0 + int(10 * k)), name, font=F, fill=(255, 220, 120))
        can.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (x0, y0 + lab))
    y += rows * (th_ + lab + 16)
    sw = [('FIRE RED', (225, 40, 40)), ('WATER BLUE', (30, 150, 235)), ('EARTH GREEN', (70, 200, 60)),
          ('AIR GREY', (200, 200, 200)), ('GOLD RULE', (206, 172, 98))]
    for i, (n, col) in enumerate(sw):
        d.rectangle((16 + i * int(300 * k), y, 16 + i * int(300 * k) + int(60 * k), y + int(60 * k)), fill=col)
        d.text((16 + i * int(300 * k) + int(72 * k), y + int(18 * k)), n, font=F, fill=(230, 230, 230))
    d.text((16, y + int(80 * k)), 'SOUND ARC  1-3 calm  |  4-5 unease  |  6 riser 2900  |  0.25 s silence  |  7 impact + sub-drop '
                         '(biggest hit)  |  8 + END loudest, sting on PLAY QUESS', font=F, fill=(220, 220, 220))
    can.save(out, quality=90)
    print('wrote', out, can.size)
