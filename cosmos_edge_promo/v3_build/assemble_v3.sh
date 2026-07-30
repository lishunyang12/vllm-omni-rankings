#!/bin/bash
set -e
G=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen
C=$G/cards; V=$G/vo; S=$G/segs; mkdir -p "$S"
FPS=30
# segment durations (VO + padding)
d1=9.22; d2=8.60; d3=4.27; d4=4.96; d5=8.10; d6=6.35
# slowdown factors so demo motion fills the caption time
f2=1.091; f3=1.103; f4=1.282
COV="scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"

echo "== seg1 title =="
ffmpeg -y -loop 1 -i "$C/title.png" -t $d1 -r $FPS \
  -vf "fade=t=in:st=0:d=0.4,fade=t=out:st=$(echo "$d1-0.4"|bc):d=0.4,format=yuv420p" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/s1.mp4" -loglevel error

seg_demo(){ # $1 src  $2 cap  $3 dur  $4 factor  $5 out
  ffmpeg -y -i "$1" -i "$2" -filter_complex \
   "[0:v]${COV},setpts=${4}*PTS,fps=$FPS[bg];[bg][1:v]overlay=0:0,format=yuv420p[v]" \
   -map "[v]" -t "$3" -c:v libx264 -crf 18 -pix_fmt yuv420p "$5" -loglevel error; }

echo "== seg2 hero I2V driving =="
seg_demo "$G/v3_i2v_driving.mp4" "$C/cap_hero.png"   $d2 $f2 "$S/s2.mp4"
echo "== seg3 T2V sort =="
seg_demo "$G/v3_t2v_sort.mp4"    "$C/cap_sort.png"   $d3 $f3 "$S/s3.mp4"
echo "== seg4 action umi =="
seg_demo "$G/v3_i2v_umi.mp4"     "$C/cap_action.png" $d4 $f4 "$S/s4.mp4"

echo "== seg5 fp8 side-by-side =="
ffmpeg -y -stream_loop 4 -i "$G/edge_t2v_official.mp4" -stream_loop 4 -i "$G/edge_t2v_official_fp8.mp4" -i "$C/cap_fp8.png" \
 -filter_complex \
 "[0:v]scale=640:-2,pad=640:720:0:(720-ih)/2:color=black,fps=$FPS[l];\
  [1:v]scale=640:-2,pad=640:720:0:(720-ih)/2:color=black,fps=$FPS[r];\
  [l][r]hstack=inputs=2[s];[s][2:v]overlay=0:0,format=yuv420p[v]" \
 -map "[v]" -t $d5 -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/s5.mp4" -loglevel error

echo "== seg6 close =="
ffmpeg -y -loop 1 -i "$C/close.png" -t $d6 -r $FPS \
  -vf "fade=t=in:st=0:d=0.4,fade=t=out:st=$(echo "$d6-0.5"|bc):d=0.5,format=yuv420p" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/s6.mp4" -loglevel error

echo "== concat video =="
printf "file '%s'\n" "$S/s1.mp4" "$S/s2.mp4" "$S/s3.mp4" "$S/s4.mp4" "$S/s5.mp4" "$S/s6.mp4" > "$S/list.txt"
ffmpeg -y -f concat -safe 0 -i "$S/list.txt" -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/video.mp4" -loglevel error

echo "== build VO audio track (each VO padded to its segment, 0.15s lead) =="
mkva(){ ffmpeg -y -i "$1" -af "adelay=150|150,apad" -t "$2" -ar 44100 -ac 2 "$3" -loglevel error; }
mkva "$V/s1_title.wav"  $d1 "$S/a1.wav"
mkva "$V/s2_hero.wav"   $d2 "$S/a2.wav"
mkva "$V/s3_t2v.wav"    $d3 "$S/a3.wav"
mkva "$V/s4_action.wav" $d4 "$S/a4.wav"
mkva "$V/s5_fp8.wav"    $d5 "$S/a5.wav"
mkva "$V/s6_close.wav"  $d6 "$S/a6.wav"
printf "file '%s'\n" "$S/a1.wav" "$S/a2.wav" "$S/a3.wav" "$S/a4.wav" "$S/a5.wav" "$S/a6.wav" > "$S/alist.txt"
ffmpeg -y -f concat -safe 0 -i "$S/alist.txt" -c:a aac -b:a 192k "$S/audio.m4a" -loglevel error

echo "== mux =="
OUTDIR=/home/zjy/code/lsy/vllm-omni-rankings/cosmos_edge_promo
ffmpeg -y -i "$S/video.mp4" -i "$S/audio.m4a" -c:v copy -c:a aac -shortest \
  "$OUTDIR/cosmos_edge_promo_v3.mp4" -loglevel error
echo "DONE -> $OUTDIR/cosmos_edge_promo_v3.mp4"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -show_entries format=duration -of default=nw=1 "$OUTDIR/cosmos_edge_promo_v3.mp4"
