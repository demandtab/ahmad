#!/bin/bash
# Inventory one Drive video: download, probe, 1 frame / 2 s contact sheet, 540p proxy, delete original.
# usage: inventory.sh <drive_id> <title> <workdir>
set -u
id="$1"; title="$2"; W="$3"
safe=$(echo "$title" | tr ' /' '__')
f="$W/dl/$safe"
[ -s "$W/probe/$safe.txt" ] && exit 0
mkdir -p "$W/dl" "$W/probe" "$W/sheets" "$W/proxy"
for n in 1 2 3 4; do
  curl -sSL -o "$f" "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t" && break; sleep $((2**n))
done
ffprobe -v error -show_entries format=duration,size:stream=codec_type,width,height,r_frame_rate -of compact "$f" > "$W/probe/$safe.tmp"
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
n=$(python3 -c "import math;print(max(1,math.ceil($dur/2)))")
cols=6; rows=$(( (n + cols - 1) / cols ))
ffmpeg -nostdin -v error -y -i "$f" -vf "fps=0.5,scale=320:-2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='%{pts\:hms}':x=4:y=4:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6,tile=${cols}x${rows}" -frames:v 1 "$W/sheets/$safe.jpg"
ffmpeg -nostdin -v error -y -i "$f" -vf "scale='if(gt(iw,ih),-2,540)':'if(gt(iw,ih),540,-2)'" -c:v libx264 -crf 26 -preset veryfast -an "$W/proxy/$safe"
mv "$W/probe/$safe.tmp" "$W/probe/$safe.txt"
rm -f "$f"
echo "done $title"
