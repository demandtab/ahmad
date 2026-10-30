"""Render the reel cut (scenes_reel). Same CLI as render_pm.py."""
import os
import sys

os.environ['SCENES_MOD'] = 'scenes_reel'
import render_pm  # noqa: E402

if __name__ == '__main__':
    if sys.argv[1] == 'sheet':
        render_pm.sheet([float(x) for x in sys.argv[3:]], sys.argv[2])
    else:
        a = sys.argv[2:]
        render_pm.video(a[0], int(a[1]) if len(a) > 1 else 8, int(a[2]) if len(a) > 2 else 16)
