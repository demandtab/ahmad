# Quess trailer, locked recipe: reply

Render: `scenes_lock.py` / `render_lock.py` / `mix_lock.py`, built layer-for-layer from the locked beat table.
Preview sent: `Quess_Trailer_Locked_360p.mp4` (360x640, 35.00 s), as requested. **The final export is 1080x1920 only:** `SC=0.5 python3 render_lock.py video out.mp4`.

## Layers that could not match the lock exactly (reported, not substituted)
| Layer | Locked spec | Why it fails | What was rendered |
|---|---|---|---|
| Headlines | size 100 (one line) / 88 (two lines) | At 100/88 px, only "THEY JOIN" fits one line inside the 800 px safe width. The others need 2–4 lines at 88. | Sizes kept exactly at 100/88 and wrapped. Each block is centred on y=330 and moved down only when it would enter the top 220 px. |
| text "CH-EAAF79" | size 140 | 914 px wide; crosses the right 140 px safe zone | size 122 (796 px) |
| text "QUESS - 4 PLAYER CHESS" | size 58 | 864 px wide | size 53 |
| player labels | one line at x=330 / 750 | "JOHN GREEN EARTH GREEN" is 565 px; it crosses the safe zone and overlaps the TELLER label | two lines (name / army) for all four labels, same size 34, same positions |
| room_code_panel.png | panel top at y≈808, code at y=820 | the panel's "Finding Quess Match" title sits under the code | panel cropped just below that title (real "1/4 Players Matched" panel with the code field) |
| player → army mapping | TELLER FIRE RED, JOHN GREEN EARTH GREEN, JACKY WATER BLUE, OPIE AIR GREY | The recordings don't fix this: the HUD bar colour follows the viewer, and John Green shows in both the red and grey bars. | rendered as written; **please confirm** |
| music | generated 35 s track | no music generator in this environment | Mixkit "Epic Games" (#76), beat-cut to the locked map (see AUDIO). Replace it with the generated track when you have it. |

## [ASSETS] (all real; cropped from the Drive recordings and the real outro)
- board_lowangle.png: Game Quess Env.mp4 at 46 s. Low angle across WATER BLUE and AIR GREY armies, 4-player board, 3840x2160.
- private_room.png: recording.mp4 at 14.5 s. Real Private Room panel: Host or Join, Create Room / Join Room. No Hotseat.
- choose_game.png: recording.mp4 at 5.0 s. Real Choose Game To Play panel with CheQuers selected; Quess and BaQugammon beside it.
- room_code_panel.png: recording.mp4 at 45 s. Real Finding Quess Match panel: 1/4 Players Matched, globe, code field CH-EAAF79.
- card_teller_fire_red.png: Movie_079.mp4 at 4 s. FIRE RED elemental player on the throne.
- card_john_green_earth_green.png: Movie_079.mp4 at 14.5 s. EARTH GREEN player on the throne.
- card_jacky_water_blue.png: Movie_079.mp4 at 8 s. WATER BLUE player on the throne.
- card_opie_air_grey.png: Movie_079.mp4 at 11 s. AIR GREY player on the throne.
- board_capture.png: Quess Gameplay tra.mp4 at 5.55 s. AIR GREY captures a FIRE RED piece; the burst is just starting.
- board_topdown.png: Game Quess Env.mp4 at 35 s. Top-down 4-player board between the WATER BLUE and FIRE RED players.
- quess_q_logo.png: outro.mp4 at 1.4 s. The full silver Q, keyed, uncropped (869x1374 inside the frame).
- company_logo.png, available_on.png, store_badges.png, url.png: outro.mp4 at 4.2 s, cropped exactly. Quess Gaming Company wordmark, AVAILABLE ON, Google Play / App Store / STEAM, www.quessgaming.com.
- bg_stars.png: outro.mp4 starfield (z=0), with gold sparkles at z=0 only.

## [RECIPE]
```
00:00.00 -> 00:35.00 image bg_stars.png
pos=540,960 w=1.0 z=0 in=cut out=cut
00:00.00 -> 00:30.00 shape "gold sparkles"
pos=540,960 w=1.0 z=0 color=#C9A24A
00:00.00 -> 00:03.00 image board_lowangle.png
pos=540,960 w=3.16 z=2 in=fade:0.25 out=cut
00:00.00 -> 00:03.00 camera "push"
zoom=1.00->1.08
00:00.00 -> 00:03.00 text "WHEN YOU FINALLY FIND THE PERFECT MOVE..."
pos=540,330 z=5 font=Cinzel weight=900 size=88 color=#FFFFFF align=center in=scale:1.08->1.0:0.25 out=cut
00:00.00 -> 00:03.00 shape "underline"
pos=540,UNDER w=0.30 size=4 color=#C9A24A z=5 in=scale:1.08->1.0:0.25 out=cut
00:03.00 -> 00:07.00 image private_room.png
pos=540,980 w=0.94 z=2 in=tilt:18->0:1.2 out=cut
00:04.20 -> 00:07.00 camera "zoom panel"
zoom=1.00->1.10
00:03.00 -> 00:07.00 text "CREATE A PRIVATE ROOM"
pos=540,330 z=5 font=Cinzel weight=900 size=88 color=#FFFFFF align=center in=scale:1.08->1.0:0.25 out=cut
00:07.00 -> 00:11.00 image choose_game.png
pos=540,940 w=0.94 z=2 in=tilt:-16->0:1.2 out=cut
00:08.20 -> 00:11.00 camera "push"
zoom=1.00->1.05
00:07.00 -> 00:11.00 text "CHOOSE YOUR GAME"
pos=540,330 z=5 font=Cinzel weight=900 size=88 align=center in=scale:1.08->1.0:0.25 out=cut
00:08.50 -> 00:11.00 text "CheQuers"
pos=540,1400 z=5 font=Cinzel weight=900 size=100 color=#FFFFFF align=center in=scale:1.3->1.0:0.3 out=cut
00:11.00 -> 00:15.00 text "CH-EAAF79"
pos=540,820 z=4 font=Cinzel weight=900 size=122 color=#FFFFFF align=center in=scale:1.3->1.0:0.3 out=cut
00:11.00 -> 00:15.00 image room_code_panel.png
pos=540,1180 w=0.80 z=2 in=fade:0.3 out=cut
00:11.00 -> 00:15.00 camera "zoom into code"
zoom=1.00->1.12 center=540,820
00:11.00 -> 00:15.00 text "GET YOUR ROOM CODE"
pos=540,330 z=5 font=Cinzel weight=900 size=88 align=center in=scale:1.08->1.0:0.25 out=cut
00:15.00 -> 00:19.00 text "THEY JOIN"
pos=540,330 z=5 font=Cinzel weight=900 size=100 align=center in=scale:1.08->1.0:0.25 out=cut
00:15.30 -> 00:19.00 image card_teller_fire_red.png
pos=330,820 w=0.36 z=2 in=pop:0.25 out=cut
00:15.30 -> 00:19.00 text "TELLER\nFIRE RED"
pos=330,1035 z=5 font=Cinzel weight=900 size=34 align=center in=pop:0.25 out=cut
00:15.90 -> 00:19.00 image card_john_green_earth_green.png
pos=750,820 w=0.36 z=2 in=pop:0.25 out=cut
00:15.90 -> 00:19.00 text "JOHN GREEN\nEARTH GREEN"
pos=750,1035 z=5 font=Cinzel weight=900 size=34 align=center in=pop:0.25 out=cut
00:16.50 -> 00:19.00 image card_jacky_water_blue.png
pos=330,1250 w=0.36 z=2 in=pop:0.25 out=cut
00:16.50 -> 00:19.00 text "JACKY\nWATER BLUE"
pos=330,1465 z=5 font=Cinzel weight=900 size=34 align=center in=pop:0.25 out=cut
00:17.10 -> 00:19.00 image card_opie_air_grey.png
pos=750,1250 w=0.36 z=2 in=pop:0.25 out=cut
00:17.10 -> 00:19.00 text "OPIE\nAIR GREY"
pos=750,1465 z=5 font=Cinzel weight=900 size=34 align=center in=pop:0.25 out=cut
00:19.00 -> 00:21.50 image board_capture.png
pos=918,960 w=2.84 z=2 in=cut out=cut
00:19.00 -> 00:21.50 camera "fast push"
zoom=1.00->1.15
00:20.75 -> 00:21.00 shape "capture burst"
pos=540,922 w=0.30 color=#FF6A2A z=3 in=flash:0.08 out=fade:0.17
00:19.00 -> 00:21.50 text "LET THE GAME BEGIN"
pos=540,330 z=5 font=Cinzel weight=900 size=88 align=center in=scale:1.08->1.0:0.25 out=cut
00:21.50 -> 00:24.00 image board_lowangle.png
pos=1180,1060 w=4.4 z=2 in=cut out=cut
00:21.50 -> 00:24.00 camera "push"
zoom=1.00->1.12
00:21.50 -> 00:24.00 text "THEN PLAYER 2 MOVES. THEN PLAYER 3 MOVES."
pos=540,330 z=5 font=Cinzel weight=900 size=88 align=center in=scale:1.08->1.0:0.25 out=cut
00:24.00 -> 00:28.00 image board_topdown.png
pos=540,960 w=3.16 z=2 in=cut out=cut
00:24.00 -> 00:26.00 text "YOUR PERFECT MOVE JUST GOT COMPLICATED."
pos=540,330 z=5 font=Cinzel weight=900 size=88 align=center in=scale:1.08->1.0:0.25 out=cut
00:26.00 -> 00:28.00 text "THREE OPPONENTS."
pos=540,330 z=5 font=Cinzel weight=900 size=88 align=center in=scale:1.08->1.0:0.25 out=cut
00:26.00 -> 00:26.35 shape "edge flash"
pos=540,960 w=1.0 color=#1E96F0 z=3 in=flash:0.05 out=fade:0.3
00:26.40 -> 00:26.75 shape "edge flash"
pos=540,960 w=1.0 color=#D7D7D7 z=3 in=flash:0.05 out=fade:0.3
00:26.80 -> 00:27.15 shape "edge flash"
pos=540,960 w=1.0 color=#46D23C z=3 in=flash:0.05 out=fade:0.3
00:28.00 -> 00:30.00 image quess_q_logo.png
pos=540,900 w=0.55 z=4 in=scale:0.6->1.0:0.8 out=fade:0.2
00:30.00 -> 00:35.00 text "QUESS - 4 PLAYER CHESS"
pos=540,560 z=5 font=Cinzel weight=900 size=53 align=center in=fade:0.3 out=none
00:30.00 -> 00:35.00 image company_logo.png
pos=540,780 w=0.70 z=4 in=fade:0.3 out=none
00:30.00 -> 00:35.00 image available_on.png
pos=540,990 w=0.28 z=4 in=fade:0.3 out=none
00:30.00 -> 00:35.00 image store_badges.png
pos=540,1090 w=0.76 z=4 in=fade:0.3 out=none
00:30.00 -> 00:35.00 image url.png
pos=540,1210 w=0.42 z=4 in=fade:0.3 out=none
```
Every headline gets the same underline layer (w=0.30, size=4, #C9A24A, z=5) directly under its last line, with the same times as the headline. The block above shows it once, for the first headline.

## [AUDIO]
```
00:00.00 -> 00:35.00 music "Epic Games (Mixkit #76), beat-edited"
levels: 00:00-00:03 full (full-energy bar, drums from frame one) | 00:03-00:15 the track's build | 00:15-00:24 full |
        00:24-00:28 climax +2.5 dB, the track's drop exactly at 00:24.00 | 00:28-00:30 -6 dB under the logo |
        00:30-00:35 stays -6 dB, fade to zero by 00:34.50
00:03.00 sfx "soft UI click" (Mixkit #900), Private Room appears
00:15.30 sfx "soft pop" (Mixkit #2356 Dry pop), first card only
00:20.75 sfx "capture hit" (Mixkit #2150), on the burst, under the music
00:28.00 sfx "soft logo chime" (Mixkit #3109), nothing under it
mix: -14.0 LUFS integrated, true peak -1.5 dBTP, no clipping
```
Measured RMS by section (dBFS):

| Section | 0–3 | 3–9 | 9–15 | 15–20 | 20–24 | 24–28 | 28–30 | 30–33.5 | 33.5–34.5 | after 34.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| RMS | −16.8 | −18.8 | −16.7 | −16.1 | −17.7 | **−13.9** | −22.7 | −22.7 | −29.7 | silent |

## [SELF-CHECK]
- **F1 PASS.** Every headline is at y=330 at the top. Sparkles are at z=0, drawn before the heroes; text is z=5 and drawn last. Nothing is over text. Example: 00:01.00, "WHEN YOU…" at the top over the board.
- **F2 PASS.** The headline track is continuous from 00:00 to 00:28 with nine headlines, no gaps and no textless beats. Example: 00:05.00 is "CREATE A PRIVATE ROOM" on the panel.
- **F3 PASS.** Every headline is out=cut and the next one starts on the same frame. Only one headline is ever drawn (code: `HEADLINES` loop, `t0 <= t < t1`). Example: 00:12.13 shows only "GET YOUR ROOM CODE".
- **F4 PASS.** Each beat has one hero change. 19.00–21.50 is a single capture board; 24.00–28.00 is one top-down layer. There is no piece montage.
- **F5 PASS.** The logo is the full Q, uncropped, at w=0.55 centred on 540,900. It spans y≈448–1352, fully inside the frame (00:28.40 and 00:29.50 checked).
- **F6 PASS.** Private Room (03–07), Choose Game with CheQuers (07–11), Room Code CH-EAAF79 (11–15) and They Join (15–19) are all present.
- **F7 — NOT YET.** The 360p was sent because you asked for it first. The final export is 1080x1920 / 30 fps / 35.00 s and will be rendered after your OK.
- **F8 PASS (style), with a caveat.** The music is energetic from the first frame (a full-energy bar, not a drone) and heroic drums plus strings, with the drop on 24.0. The track is a stand-in; the generated track should replace it.
