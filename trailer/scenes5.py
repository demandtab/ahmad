"""FINAL videos.

  FINAL=mobile   V1 shots and look (phones), revision-3 fonts + THEY JOIN lobby
  FINAL=desktop  same flow on the real filmed gaming-PC setup (Mixkit green-screen footage), revision-3 fonts + lobby

Both: original private-room script order, no Hotseat, uploaded logos and outro, Mixkit "Epic Games".
"""
import os

FINAL = os.environ.get('FINAL', 'mobile')
if FINAL == 'desktop':
    os.environ['R3MODE'] = 'desktop'
    os.environ['VARIANT'] = 'v2'
else:
    os.environ['VARIANT'] = 'v1'

from common import clip, eo3, lerp, seg  # noqa: E402
from pc import pc_frame  # noqa: E402
import scenes3 as s3  # noqa: E402

DURATION = s3.DURATION

if FINAL == 'desktop':
    _choose = s3._M['choose']

    def _choose_desktop(t):
        if t < 8.72:   # open on the real gaming setup itself (replaces the pink-room establishing clip)
            lt = t - 8.15
            ui = clip('rec_select').at(0.2 + lt * 0.5)
            f, _ = pc_frame('pc51602', 4.4 + lt, ui, zoom=lerp(0.9, 0.98, lt / 0.57), focus=(0.44, 0.46))
            if t < 8.35:
                f += 1.0 * (1 - eo3(seg(t, 8.15, 8.35)))
            return f
        return _choose(t)

    s3._M['choose'] = _choose_desktop
    import scenes as old  # noqa: E402
    from common import remap  # noqa: E402
    old.VARIANT = 'v1'   # the gameplay finale renders full screen

    def _begin_fullscreen(t):
        return old.s_begin(remap(t, [(23.0, 23.0), (24.15, 23.85), (24.72, 24.6), (25.29, 25.35), (25.86, 26.0)]))

    s3.k_begin = _begin_fullscreen
    s3.k_friends = s3._M['friends']
    render = s3.render
else:
    import scenes4 as base   # noqa: E402
    base.CAPTIONS = [c if c[0] != ('TRY IT NOW',) else (c[0], c[1], c[2], 1920, c[4], c[5]) for c in s3.CAPTIONS]
    base.CAPTIONS[0] = base.CAPTIONS[0][:5] + (False,)   # GOT THREE FRIENDS?: clean type, no glass
    base.k_end = s3.k_end
    render = base.render
