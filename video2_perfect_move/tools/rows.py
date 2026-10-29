"""Stack several contact sheets (first row of each, labelled) into one review image.
usage: python3 rows.py out.jpg sheet1.jpg sheet2.jpg ..."""
import sys
from PIL import Image, ImageDraw, ImageFont
out, files = sys.argv[1], sys.argv[2:]
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
rows = []
for f in files:
    im = Image.open(f)
    rows.append((f.split('/')[-1][:-4], im))
W = max(im.width for _, im in rows)
H = sum(im.height + 26 for _, im in rows)
canvas = Image.new('RGB', (W, H), (20, 20, 20))
y = 0
d = ImageDraw.Draw(canvas)
for name, im in rows:
    d.text((6, y + 3), name, font=F, fill=(255, 220, 80))
    canvas.paste(im, (0, y + 26))
    y += im.height + 26
canvas.save(out, quality=85)
