#!/bin/bash
set -e
G=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen
C=$G/cards; V=$G/vo4; S=$G/segs4; mkdir -p "$S"
FPS=30
d1=9.38; d2=7.70; d3=4.12; d4=4.08; d5=7.12; d6=5.04
COV="scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"

ffmpeg -y -loop 1 -i "$C/title.png" -t $d1 -r $FPS \
  -vf "fade=t=in:st=0:d=0.4,fade=t=out:st=$(echo "$d1-0.4"|bc):d=0.4,format=yuv420p" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/s1.mp4" -loglevel error

# demo segment: cover-scale, freeze last frame to fill (tpad), overlay caption, cut to dur
seg_demo(){ ffmpeg -y -i "$1" -i "$2" -filter_complex \
   "[0:v]${COV},fps=$FPS,tpad=stop_mode=clone:stop_duration=12[bg];[bg][1:v]overlay=0:0,format=yuv420p[v]" \
   -map "[v]" -t "$3" -c:v libx264 -crf 18 -pix_fmt yuv420p "$4" -loglevel error; }

echo "seg2 hero coastal"; seg_demo "$G/repro_official_i2v.mp4" "$C/cap_hero.png"      $d2 "$S/s2.mp4"
echo "seg3 warehouse";    seg_demo "$G/v4_t2v_warehouse.mp4"   "$C/cap_warehouse.png" $d3 "$S/s3.mp4"
echo "seg4 robot";        seg_demo "$G/v4_t2v_robot.mp4"       "$C/cap_robot.png"     $d4 "$S/s4.mp4"

echo "seg5 fp8 split (coastal dense | fp8, same seed)"
ffmpeg -y -i "$G/repro_official_i2v.mp4" -i "$G/v5_coastal_fp8.mp4" -i "$C/cap_fp8.png" \
 -filter_complex \
 "[0:v]scale=640:-2,pad=640:720:0:(720-ih)/2:color=black,fps=$FPS[l];\
  [1:v]scale=640:-2,pad=640:720:0:(720-ih)/2:color=black,fps=$FPS[r];\
  [l][r]hstack=inputs=2,tpad=stop_mode=clone:stop_duration=12[s];[s][2:v]overlay=0:0,format=yuv420p[v]" \
 -map "[v]" -t $d5 -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/s5.mp4" -loglevel error

echo "seg6 close"
ffmpeg -y -loop 1 -i "$C/close.png" -t $d6 -r $FPS \
  -vf "fade=t=in:st=0:d=0.4,fade=t=out:st=$(echo "$d6-0.5"|bc):d=0.5,format=yuv420p" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/s6.mp4" -loglevel error

printf "file '%s'\n" "$S/s1.mp4" "$S/s2.mp4" "$S/s3.mp4" "$S/s4.mp4" "$S/s5.mp4" "$S/s6.mp4" > "$S/list.txt"
ffmpeg -y -f concat -safe 0 -i "$S/list.txt" -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/video.mp4" -loglevel error

mkva(){ ffmpeg -y -i "$1" -af "adelay=150|150,apad" -t "$2" -ar 44100 -ac 2 "$3" -loglevel error; }
mkva "$V/s1.wav" $d1 "$S/a1.wav"; mkva "$V/s2.wav" $d2 "$S/a2.wav"; mkva "$V/s3.wav" $d3 "$S/a3.wav"
mkva "$V/s4.wav" $d4 "$S/a4.wav"; mkva "$V/s5.wav" $d5 "$S/a5.wav"; mkva "$V/s6.wav" $d6 "$S/a6.wav"
printf "file '%s'\n" "$S/a1.wav" "$S/a2.wav" "$S/a3.wav" "$S/a4.wav" "$S/a5.wav" "$S/a6.wav" > "$S/alist.txt"
ffmpeg -y -f concat -safe 0 -i "$S/alist.txt" -c:a aac -b:a 192k "$S/audio.m4a" -loglevel error

OUTDIR=/home/zjy/code/lsy/vllm-omni-rankings/cosmos_edge_promo
ffmpeg -y -i "$S/video.mp4" -i "$S/audio.m4a" -c:v copy -c:a aac -shortest "$OUTDIR/cosmos_edge_promo_v4.mp4" -loglevel error
echo "DONE -> $OUTDIR/cosmos_edge_promo_v4.mp4"
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUTDIR/cosmos_edge_promo_v4.mp4"
