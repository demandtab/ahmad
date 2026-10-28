# Quess — vertical cinematic trailer (9:16, 4K, 34 s)

Revision 3 (`scenes3.py`, `mix3.py`): follows the original script scene by scene; Hotseat removed
everywhere (the private-room beat uses the real Private Room and Create Room screens only); real
filmed gaming-PC shots from Mixkit green-screen clips with the recordings keyed and tracked onto the
monitor (`track.py`, `pc.py`); redone THEY JOIN lobby; Cinzel/Montserrat type with a single frosted
glass title; uploaded Quess / CheQuers / BaQugammon logos and the uploaded outro as the end card.

    SC=1 VARIANT=r4m R3MODE=mobile SCENES=scenes3 python3 build_variant.py    # phone version
    SC=1 VARIANT=r4d R3MODE=desktop SCENES=scenes3 python3 build_variant.py   # desktop version (real PC footage only)
    python3 mix3.py audio.wav

Stock footage (Mixkit free licence): #51602 and #51614 (green-screen gaming setups), #51622 (gaming room).

Revision 2: premium captions (clean white condensed type, one thin gold accent line, soft
shadow; no glow, neon or gradient on any text), cut re-synced to the beat grid of Mixkit
"Epic Games" (the only music), and two variations:

- `VARIANT=v1`: phone mockups, except the Hotseat shot, which plays on a gaming PC (Hotseat is PC-only).
- `VARIANT=v2`: every UI and gameplay shot plays on the gaming PC setup; no phones anywhere.

A scripted build of the Quess trailer. All UI and gameplay comes from the real
recordings and game cinematics. The scripts add the 3D phone, arena, particles,
titles, grade and sound design on top.

## Timeline

| Time | Beat | Footage used |
|---|---|---|
| 0:00–0:03 | GOT THREE FRIENDS? — two armies face off, board opens to all four | `Game Quess Env.mp4` 57–60 s |
| 0:03–0:07 | CHOOSE YOUR GAME — CheQuers (green), BaQugammon (blue), Quess (red, last and brightest) ignite | `recording.mp4` 1.6–5.3 s in a floating phone |
| 0:07–0:12 | CREATE A PRIVATE ROOM — Private Room → Create Room, glowing envelope, slow-motion click + flash | `recording.mp4` 8.4–39.7 s |
| 0:12–0:15 | GET YOUR ROOM CODE — CH-EAAF79 lifts off the screen as gold particles, one character at a time | `recording.mp4` 39.7–42 s |
| 0:15–0:18 | SHARE IT WITH YOUR FRIENDS — code bursts to three friend avatars | avatars cut from `Movie_078.mp4` |
| 0:18–0:21 | THEY JOIN — real lobby (1/4 matched), then the four elemental players take their seats | `recording.mp4`, `Movie_078.mp4` 2–5.8 s |
| 0:21–0:23 | EVERYONE'S IN — overhead hero shot of the full 4-player board, board glow cut | `Game Quess Env.mp4` 32–35 s |
| 0:23–0:26 | LET THE GAME BEGIN — capture-smoke wipe, fast 4-player gameplay cuts | `Quess Gameplay tra.mp4`, `Quess gameplay (1).mp4`, env |
| 0:26–0:30 | PLAY WITH YOUR FRIENDS / TRY IT NOW | `Family Playing.mp4` intercut with gameplay |
| 0:30–0:34 | Logo assembles from particles, QUESS - 4 PLAYER CHESS, store badges, URL, connection note | logo/badges/URL keyed from `shot-10.mp4` |

The real app flow is choose game → create room, so the trailer follows that
order, which swaps the brief's 0:03 and 0:07 beats.

## Third-party audio

All from Mixkit under the [Mixkit free licence](https://mixkit.co/license/), no attribution required:
- Music: "Epic Games" (Mixkit #76). The `alarm` mix uses the uploaded "Alarm - Music Mix" instead.
- Sound effects: Mixkit #1489, 1491, 2596 (whooshes), 2150, 2908, 788, 2182 (impacts),
  2900 (riser), 900, 1113 (clicks), 1108, 3109 (chimes), 368, 364 (crowd).
- The low sub-drop is synthesised in `mix.py`.


## Build

```bash
pip install opencv-python-headless numpy pillow scipy
# extract source segments into ../src (see SOURCES in render notes), fonts into ../fonts
SC=0.25 python3 render.py sheet preview.jpg 1.0 5.0 12.5    # quick look
SC=1 VARIANT=v1 python3 build_variant.py                    # 4K picture, variation 1
SC=1 VARIANT=v2 python3 build_variant.py                    # 4K picture, variation 2
python3 mix.py audio.wav                                    # soundtrack (shared by both)
```

Fonts: Anton, Cinzel and Oswald (Google Fonts, SIL Open Font License).
