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


def narrator(f, lines, cy, size=150):
    img, _ = C.title_img(tuple(lines), size)
    C.place(f, img, P(CX), P(cy))


def thought(f, text, cy, size=150):
    """Inner thought: bold italic serif, sentence case, soft dark backing (no glass, no glow)."""
    fnt = ImageFont.truetype(os.path.join(C.FONTS, 'CormorantGaramond-BoldItalic.ttf'), int(P(size)))
    b = ImageDraw.Draw(Image.new('L', (1, 1))).multiline_textbbox((0, 0), text, font=fnt, align='center',
                                                                  spacing=int(P(size * 0.12)))
    pad = int(P(70))
    w, h = int(b[2] - b[0]) + 2 * pad, int(b[3] - b[1]) + 2 * pad
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).multiline_text((pad - b[0], pad - b[1]), text, font=fnt, fill=255, align='center',
                                      spacing=int(P(size * 0.12)))
    m = np.asarray(im).astype(np.float32) / 255
    back = np.zeros_like(m)
    cv2.rectangle(back, (int(pad * 0.5), int(pad * 0.55)), (w - int(pad * 0.5), h - int(pad * 0.55)), 1, -1)
    back = cv2.GaussianBlur(back, (0, 0), P(34)) * 0.62
    sh = cv2.GaussianBlur(np.roll(m, int(P(5)), 0), (0, 0), P(5)) * 0.7
    a = np.clip(np.maximum(back, sh) + m, 0, 1)
    rgb = np.dstack([m] * 3) * 0.97
    rgb = np.where(a[..., None] > 0, rgb / np.maximum(a[..., None], 1e-4), 0)
    C.place(f, np.dstack([rgb, a]).astype(np.float32), P(CX), P(cy))


def frame(name, focus, zoom, t, extra=None, narr=None, th=None, flash=None):
    f = load(name, focus, zoom)
    if extra:
        f = extra(f)
    if flash:
        f = edge_flash(f, ARMY[flash])
    f = C.grade(f, t, sat=1.15, leak=0.06, vig=0.5, grain=0.012, bloom=0.35)
    if narr:
        narrator(f, *narr)
    if th:
        thought(f, *th)
    return C.to_u8(f)


TILES = [
    ('1 HOOK  push-in, tension', frame('hook', (0.47, 0.55), 1.15, 1.0,
                                       narr=(('WHEN YOU FINALLY FIND', 'THE PERFECT MOVE...'), 760, 120))),
    ('2 THOUGHT  FIRE RED POV, spotlight', frame('red_pov', (0.55, 0.6), 1.35, 3.0,
                                                 extra=lambda f: spotlight(f, 1080, 1700, 520), flash='FIRE RED',
                                                 th=('I know exactly\nwhat to do.', 2700, 190))),
    ('3 MOVE PLAYED  overhead hold', frame('board_overhead', (0.5, 0.48), 1.0, 5.0,
                                           narr=(('MOVE PLAYED.',), 760, 150))),
    ('4-5 PLAYER 2  WATER BLUE low drift', frame('low_blue', (0.42, 0.62), 1.2, 7.0, flash='WATER BLUE',
                                                 narr=(('THEN PLAYER 2 MOVES.',), 760, 120),
                                                 th=('Ok... still fine.', 2750, 190))),
    ('6-7 IMPACT  sharp, no blur', frame('fire_char', (0.47, 0.42), 1.25, 9.0, flash='EARTH GREEN',
                                         th=('Wait.\nThat changes everything.', 2650, 175))),
    ('8 CHAOS  native 9:16 hero', frame('chaos1', (0.5, 0.5), 1.0, 11.0,
                                        narr=(('YOUR PERFECT MOVE...', 'JUST GOT COMPLICATED.'), 760, 110))),
    ('8 CHAOS  board-level dolly', frame('chaos2', (0.55, 0.55), 1.0, 12.0)),
    ('8 PULL-OUT  all four armies', frame('chaos3', (0.5, 0.55), 1.0, 13.0)),
    ('END lead-in  four players', frame('thrones', (0.5, 0.55), 1.0, 14.0,
                                        narr=(('THREE OPPONENTS.',), 760, 140))),
    ('END  real outro + lines', frame('outro', (0.5, 0.5), 1.0, 15.0,
                                      narr=(('PLAY QUESS.',), 760, 170))),
]

if __name__ == '__main__':
    out = sys.argv[2]
    cols, tw, th_ = 5, W, H
    lab = 44
    F = ImageFont.truetype(os.path.join(C.FONTS, 'Montserrat-SemiBold.ttf'), 22)
    FT = ImageFont.truetype(os.path.join(C.FONTS, 'Cinzel_wght_900.ttf'), 40)
    rows = (len(TILES) + cols - 1) // cols
    ref = [Image.open(os.path.join(FR, f'ref_{i}.jpg')) for i in (1, 10, 22, 27, 30)]
    rw, rh = ref[0].size
    head = 90
    ref_h = rh + lab + 20
    sw_h = 150
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
        d.text((x0, y0 + 10), name, font=F, fill=(255, 220, 120))
        can.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (x0, y0 + lab))
    y += rows * (th_ + lab + 16)
    sw = [('FIRE RED', (225, 40, 40)), ('WATER BLUE', (30, 150, 235)), ('EARTH GREEN', (70, 200, 60)),
          ('AIR GREY', (200, 200, 200)), ('GOLD RULE', (206, 172, 98))]
    for i, (n, col) in enumerate(sw):
        d.rectangle((16 + i * 300, y, 16 + i * 300 + 60, y + 60), fill=col)
        d.text((16 + i * 300 + 72, y + 18), n, font=F, fill=(230, 230, 230))
    d.text((16, y + 80), 'SOUND ARC  1-3 calm  |  4-5 unease  |  6 riser 2900  |  0.25 s silence  |  7 impact + sub-drop '
                         '(biggest hit)  |  8 + END loudest, sting on PLAY QUESS', font=F, fill=(220, 220, 220))
    can.save(out, quality=90)
    print('wrote', out, can.size)
