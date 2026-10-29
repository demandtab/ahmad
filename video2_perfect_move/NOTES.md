# Quess Video #2 — "The Perfect Move" — working notes

- Branch: claude/great-goldberg-1j0j98 (session-assigned; based on claude/upbeat-edison-ldwkb4, trailer/ code importable).
- Source folder (Drive, link-shared): 117FuqPsIFcxsMmTPHBu_h4excnPF_9y3 "Recordings", 76 videos top level.
  Listed via https://drive.google.com/embeddedfolderview?id=... (Drive MCP search does not index it).
- Assets (logo/outro/badges): folder 1csaUAoc7SASP4kpulzVibqDZJqyYLVf4 -> outro.mp4 (1080x1920, 6.3 s: logo, AVAILABLE ON, Google Play/App Store/Steam badges, www.quessgaming.com).
- Inventory: tools/inventory.sh (download -> probe -> 1 frame/2 s sheet -> 540p proxy -> delete original). Media lives outside the repo in /home/user/media.
- Fonts (Google Fonts, OFL), Mixkit SFX/music are fetched into ../fonts, ../sfx, ../music (git-ignored).
- Reference trailer Quess_Trailer_FINAL_Mobile_1080p.mp4: NOT YET UPLOADED (asked).
- Reference trailer: Drive 10MdgKFiMvXf5u_UiH-F0kHxnWdnGyJSs (1080x1920, 34.5 s). 1fps sheet in /home/user/media/ref/ref_1fps.jpg.
- Inventory complete: 76/76 (MEDIA_INDEX.md). BLOCKER: no real move sequence in the folder; asked the user (option: old-folder "Quess gameplay (1).mp4" / "Quess Gameplay tra.mp4").
- Moodboard: moodboard.py -> moodboard_v1.jpg (real frames, trailer grade, proposed type).

## 720p preview v1 — SNAPped cut / hit times (s, frame-exact, Mixkit #614 beat grid, 0.43 s beat)
- cuts: p1 2.967, p1m 4.700, p1a 7.300, p2 8.533, p2a 11.033, p3 12.767, hit 15.400, react 17.967, l8 19.233, c097 20.533, fcp 21.800, burn 22.667, qc 23.533, end 25.700, outro 29.100
- text hits: 0.400 WHEN YOU / FINALLY FIND, 1.700 THE PERFECT / MOVE..., 3.367 I know exactly / what to do., 7.300 MOVE PLAYED., 8.533 THEN PLAYER 2 / MOVES., 11.033 Ok... still fine., 12.767 THEN PLAYER 3 / MOVES., 15.400 Wait., 16.233 That changes / everything., 19.233 YOUR PERFECT / MOVE..., 21.800 JUST GOT / COMPLICATED., 25.700 THREE / OPPONENTS., 27.367 MORE THAN / ONE MOVE / TO WATCH., 30.767 PLAY QUESS.
- stop-down 15.150-15.400; biggest hit (line 7) 15.400; second peak PLAY QUESS 30.767; duration 33.90
- audio: -13.5 LUFS integrated, true peak -1.4 dBTP; payoff (line 8 + END) loudest sections; music Silent Descent (default, pick pending)
