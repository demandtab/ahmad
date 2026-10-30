# Quess "The Perfect Move": LogiCut reply (v2 fix)

Export: **1080x1920, 30 fps, 34.00 s (1020 frames)**, H.264 High, CRF 16, yuv420p, faststart, AAC 320k. No voiceover; on-screen text only.
Built and verified by the Python + ffmpeg pipeline in `video2_perfect_move/` (`render_pm.py` for picture, `mix_pm.py` for sound).

## [ASSETS]
| ID | File (Drive) | Use |
|---|---|---|
| A1 | Game Quess Env.mp4, 18.5–25.5 s | green figure hook |
| A2 | Quess gameplay (1).mp4 (old folder), 18.8–25.2 s | FIRE RED move, in a phone mockup |
| A3 | Quess Gameplay tra.mp4 (old folder), 0–9 s | EARTH GREEN move, AIR GREY capture, cleared board, in a phone mockup |
| A4 | Quess Room.mp4 | real game room, defocused; the arena behind the phone |
| A5 | Movie_074.mp4, 0–0.3 s | FIRE RED player cut-back |
| A6 | Movie_096.mp4 · Movie_097.mp4 · French Chess pieces.mp4 · Burn down animation.mp4 · Quess Character.mp4 | climax montage / glowing piece / icy board pull-out |
| A7 | Movie_079.mp4, 17.6–22.4 s | the four players on thrones |
| A8 | outro.mp4 (asset folder) | real logo, AVAILABLE ON, Google Play / App Store / STEAM, www.quessgaming.com |
| M1 | music: ONE track, 34 s | placeholder Mixkit "Silent Descent" #614 (from 49.0 s) until the generated track arrives |
| S1–S3 | Mixkit #1113 tick · #2150 capture hit · #3109 chime | the only three SFX |
| F1 | Cinzel Black (OFL) | the only font |

## [RECIPE] (every cut is a hard cut on the 0.5 s / 120 BPM grid; one dissolve into the outro)
| Time | Shot | Text (bottom zone, centred) |
|---|---|---|
| 00:00.0–00:03.0 | A1 green figure hook: wide, push-in, low over the pieces | 0.5 WHEN YOU / FINALLY FIND · 1.5 THE PERFECT / MOVE... |
| 00:03.0–00:11.0 | A2 phone board: before (3.0), slow-motion move (4.5), MOVE PLAYED hold (7.5); A3 EARTH GREEN move (8.5) | 3.0 I know exactly / what to do. · 7.5 MOVE PLAYED. · 8.5 THEN PLAYER 2 / MOVES. |
| 00:11.0–00:15.0 | A3 wider board (11.0), low angle into the capture (12.5) | 11.0 Ok... still fine. · 12.5 THEN PLAYER 3 / MOVES. |
| 00:15.0–00:21.0 | A3 closer board: the capture lands (15.0), cleared board (16.0); A5 cut-back (19.0, no text) | 15.0 Wait. · 16.0 That changes / everything. |
| 00:21.0–00:26.0 | A6 glowing piece (21.0), montage 22.0 / 23.0 / 23.5, pull-out 24.0 | 21.0 YOUR PERFECT / MOVE... · 23.0 JUST GOT / COMPLICATED. |
| 00:26.0–00:29.0 | A7 thrones (26.0, 27.5) | 26.0 THREE / OPPONENTS. · 27.5 MORE THAN / ONE MOVE / TO WATCH. |
| 00:29.0–00:34.0 | A8 real outro (0.3 s dissolve in), lifted at 31.5 | QUESS - 4 PLAYER CHESS · An internet connection is needed to play. (required by the original brief) · 31.5 PLAY QUESS. |

## [AUDIO]
- **Music:** one single track, 00:00.00 → 00:34.00. No loop, nothing stacked on top.
- **Level map**, applied in `mix_pm.py` (`MAP`):
  - 00:00–00:03: −6 dB
  - 00:03–00:11: slow rise to 0 dB
  - 00:11–00:21: 0 dB, steady
  - 00:21–00:29: climax at 0 dB; the track's own hits fall at 21.00 and 26.00
  - 00:29–00:32: −6 dB under the end card
  - 00:32–00:33.5: fade, reaching silence by 00:33.50
- **SFX, exactly three:**
  - 00:11.00: soft piece-placement tick (#1113)
  - 00:21.00: capture hit (#2150), set under the music
  - 00:29.00: soft logo chime (#3109), with nothing under it
- **Final:** −13.6 LUFS integrated, true peak −1.5 dBTP, no clipping (two-pass loudnorm).

## Checks
1. **Build order:**
   - Measured RMS on the placeholder: 0–3 −19.8 · 3–11 rising to −17.5 · 11–21 about −17 · **21–26 −14.2 (climax, loudest)** · 26–29 −15.3 · 29–32 −23.0 (dip) · 32–33 −36.6 · after 33.5 silent. It does not end abruptly.
   - The placeholder's lift lands exactly on 21.0, but its accents at 21 and 26 are soft (about 2× the median onset strength), not real drum hits. The generated track must supply those hits; no extra SFX were added to fake them.
2. **One music track, three SFX:** confirmed in `mix_pm.py`.
3. **Export:** 1080x1920, 34.00 s, no voiceover.

## Swapping in the generated track
```bash
SC=0.5 python3 mix_pm.py audio.wav generated_track.wav 0.0
```
```bash
ffmpeg -i Quess_PerfectMove_1080p_video.mp4 -i audio.wav -c:v copy -c:a aac -b:a 320k -shortest -movflags +faststart Quess_PerfectMove_1080p.mp4
```
Tip for the music tool: add **"120 BPM"** to the prompt. At 120 BPM, 21, 26 and 29 s all fall on beats, which is the grid this cut uses.
