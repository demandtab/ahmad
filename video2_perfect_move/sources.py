"""Source segments for "The Perfect Move": cut from the real Drive recordings into frame folders.

  python3 sources.py <media_full_dir>      # writes ../src2/<name>/0001.jpg ...

Every segment is real footage. The two gameplay recordings are motion-interpolated to 60 fps
(ffmpeg minterpolate) so their slow-motion moves stay smooth; nothing is added to the picture.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC2 = os.path.join(ROOT, 'src2')

# name: (file in media/full, start s, duration s, fps, interpolate)
SEGMENTS = {
    'hook':  ('Game_Quess_Env.mp4', 18.5, 7.0, 30, False),          # EARTH GREEN character over the real board
    'g1':    ('g1.mp4', 18.8, 6.4, 60, True),                       # Quess gameplay (1): FIRE RED turn 19.0-24.6
    'tra':   ('tra.mp4', 0.0, 9.0, 60, True),                       # Quess Gameplay tra: EARTH GREEN move, AIR GREY capture, cleared board 8.5
    'react': ('Movie_074.mp4', 0.0, 1.6, 30, False),                # FIRE RED player close-up
    'c096':  ('Movie_096.mp4', 0.3, 3.2, 30, False),                # native 9:16 WATER BLUE knight
    'c097':  ('Movie_097.mp4', 9.8, 3.4, 30, False),                # board-level dolly, EARTH GREEN bishop
    'fcp':   ('French_Chess_pieces.mp4', 5.8, 2.6, 30, False),      # FIRE RED army in flames
    'burn':  ('Burn_down_animation.mp4', 5.0, 2.8, 30, False),      # piece burns down (capture)
    'qc':    ('Quess_Character.mp4', 19.8, 4.8, 30, False),         # full 4-army board orbit (pull-out)
    'thr':   ('Movie_079.mp4', 17.6, 4.8, 30, False),               # four players on thrones
    'outro': ('outro.mp4', 0.0, 6.3, 30, False),                    # real outro (asset folder)
    'room':  ('Quess_Room.mp4', 0.0, 16.0, 30, False),              # real empty Quess game room: the phone arena
    'top':   ('Game_Quess_Env.mp4', 32.5, 6.0, 30, False),          # real top-down 4-player board between WATER BLUE and FIRE RED
    'g1b':   ('g1.mp4', 25.4, 8.6, 60, True),
    'c096f': ('Movie_096.mp4', 0.0, 15.6, 30, False),               # whole portrait take: blue knight 0-4.7, green bishop 10-15
    'fcp2':  ('French_Chess_pieces.mp4', 5.6, 4.2, 30, False),      # FIRE RED pieces in flames (longer)
}


def extract(media, name):
    f, ss, dur, fps, interp = SEGMENTS[name]
    out = os.path.join(SRC2, name)
    if os.path.isdir(out) and len(os.listdir(out)) >= int(dur * fps) - 2:
        return
    os.makedirs(out, exist_ok=True)
    src = os.path.join(media, f) if name != 'outro' else os.path.join(os.path.dirname(media), 'assets', f)
    vf = f'minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1' if interp else f'fps={fps}'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(ss), '-t', str(dur), '-i', src, '-vf', vf,
                    '-q:v', '2', os.path.join(out, '%04d.jpg')], check=True)
    print('extracted', name, len(os.listdir(out)), 'frames')


if __name__ == '__main__':
    media = sys.argv[1]
    for n in (sys.argv[2:] or SEGMENTS):
        extract(media, n)
