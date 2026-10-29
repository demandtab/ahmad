# Type & text-animation spec (locked) — implemented in `type2.py`

All measurements are given at 1080x1920. The code works in 4K units, which are double these.

## Fonts
- **Narrator** (lines 1, 3, 4, 6, 8, END): Cinzel Black, ALL-CAPS, tracking 0.06 em, ivory, with a centred gold rule (GOLD RULE swatch #CEAC62).
- **Thought** (lines 2, 5, 7): Cormorant Garamond Bold Italic, sentence case, ivory, with a soft dark backing and shadow.
- Army colours never touch the letters. They appear only as the edge flash and as a light sweep behind the text zone.

## Fixed sizes (never resized from beat to beat)
- One size for every narrator line and one for every thought line: **86 px font** at 1080. Narrator capitals are about 60 px tall.
- **How this compares with the last trailer:** that trailer's captions measured about 96 px. They also ran into the right 140 px UI zone; "EVERYONE'S IN" reaches x≈985.
- **Why 86 px:** text centred on the frame has at most **800 px** of width. "COMPLICATED." can't be split, so 86 px is the largest size where every line fits. That makes it about 90% of the trailer's size, the closest match the safe zones allow.
- Lines that are too long are split into two lines at the same size, never shrunk:
  - 1: WHEN YOU / FINALLY FIND → THE PERFECT / MOVE... (one line revealed after another, at most 2 on screen)
  - 3: MOVE PLAYED.
  - 4 / 6: THEN PLAYER 2 / MOVES. · THEN PLAYER 3 / MOVES.
  - 8: YOUR PERFECT / MOVE... → JUST GOT / COMPLICATED.
  - END: the three statements build on beats, at most 3 lines on screen: THREE / OPPONENTS. → MORE THAN / ONE MOVE / TO WATCH. → PLAY QUESS. At the fixed size, MORE THAN ONE MOVE TO WATCH. fits only as three short lines. PLAY QUESS. lands with the real outro, which is shifted up as one untouched layer so its logo, badges and URL sit above the text zone.
  - 2: I know exactly / what to do. · 5: Ok... still fine. · 7: Wait. / That changes everything.
- Gold rule: fixed 190 px width, 3.5 px weight, 20 px gap under every narrator block. It is centred and tapers at its ends.

## Placement
- **One zone: BOTTOM (locked).** Every block, narrator and thought, ends at the same place: its last ink row (the gold rule, or the last line of a thought) sits 1500 px from the top, 40 px clear of the bottom 380 px UI zone. Blocks grow upward. Nothing is placed at the top or in the middle.
- Everything is centred on the frame. The maximum width of 800 px keeps text out of the right 140 px zone. Nothing goes in the top 220 px or the bottom 380 px.

## Animation (each hit SNAPped frame-exact to the beat grid)
- Narrator text reveals per line. Each line scales in from 1.15× with blur-to-sharp in about 0.2 s, with a small impact shake on the hit.
- The gold rule draws outward from the centre with a light sweep that catches it once (the "ignite" from the game panels).
- On each player's turn, an army-colour light band sweeps behind the text zone.
- Custom, footage-built reveals:
  - letters assemble from ember particles (hook and PLAY QUESS);
  - a capture-slam wipe reveals line 8;
  - text emerges from board glow (line 3).
- **Line 7** is a hard, sharp cut-in with no blur at all. Only the screen shake and the impact land on it.
- Depth: a deep vignette, plus ember and bokeh particles between the board and the type.
- None of these: glass, neon, glow on letters, flicker, pulsing.
