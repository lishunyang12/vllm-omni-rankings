#!/bin/bash
set -e
G=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen
OFF=/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_official
C=$G/cards5; V=$G/vo5; S=$G/segs5; mkdir -p "$S"
FPS=30
d1=6.9; d2=10.2; d3=7.7; d4=7.0; d5=7.8; d6=6.1; d7=9.3; d8=6.3

card(){ ffmpeg -y -loop 1 -i "$1" -t "$2" -r $FPS \
  -vf "fade=t=in:st=0:d=0.4,fade=t=out:st=$(echo "$2-0.4"|bc):d=0.4,format=yuv420p" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p "$3" -loglevel error; }

echo s1; card "$C/s1_title.png"  $d1 "$S/s1.mp4"
echo s2; card "$C/s2_caps.png"   $d2 "$S/s2.mp4"
echo s3; card "$C/s3_arch.png"   $d3 "$S/s3.mp4"
echo s4; card "$C/s4_family.png" $d4 "$S/s4.mp4"

echo s5 action
ffmpeg -y -stream_loop -1 -i "$OFF/edge_action_fd_umi_2chunk_output.mp4" -loop 1 -i "$C/s5_action_bg.png" \
 -filter_complex "[0:v]scale=452:452:flags=lanczos,fps=$FPS[c];[1:v][c]overlay=414:124,format=yuv420p[v]" \
 -map "[v]" -t $d5 -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/s5.mp4" -loglevel error

echo s6; card "$C/s6_bench.png" $d6 "$S/s6.mp4"

echo s7 fp8
ffmpeg -y -i "$G/repro_official_i2v.mp4" -i "$G/v5_coastal_fp8.mp4" -i "$C/s7_fp8.png" \
 -filter_complex \
 "[0:v]scale=640:-2,pad=640:720:0:(720-ih)/2:color=black,fps=$FPS[l];\
  [1:v]scale=640:-2,pad=640:720:0:(720-ih)/2:color=black,fps=$FPS[r];\
  [l][r]hstack=inputs=2,tpad=stop_mode=clone:stop_duration=12[s];[s][2:v]overlay=0:0,format=yuv420p[v]" \
 -map "[v]" -t $d7 -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/s7.mp4" -loglevel error

echo s8; card "$C/s8_close.png" $d8 "$S/s8.mp4"

printf "file '%s'\n" "$S/s1.mp4" "$S/s2.mp4" "$S/s3.mp4" "$S/s4.mp4" "$S/s5.mp4" "$S/s6.mp4" "$S/s7.mp4" "$S/s8.mp4" > "$S/list.txt"
ffmpeg -y -f concat -safe 0 -i "$S/list.txt" -c:v libx264 -crf 18 -pix_fmt yuv420p "$S/video.mp4" -loglevel error

mkva(){ ffmpeg -y -i "$1" -af "adelay=150|150,apad" -t "$2" -ar 44100 -ac 2 "$3" -loglevel error; }
mkva "$V/s1.wav" $d1 "$S/a1.wav"; mkva "$V/s2.wav" $d2 "$S/a2.wav"; mkva "$V/s3.wav" $d3 "$S/a3.wav"
mkva "$V/s4.wav" $d4 "$S/a4.wav"; mkva "$V/s5.wav" $d5 "$S/a5.wav"; mkva "$V/s6.wav" $d6 "$S/a6.wav"
mkva "$V/s7.wav" $d7 "$S/a7.wav"; mkva "$V/s8.wav" $d8 "$S/a8.wav"
printf "file '%s'\n" "$S/a1.wav" "$S/a2.wav" "$S/a3.wav" "$S/a4.wav" "$S/a5.wav" "$S/a6.wav" "$S/a7.wav" "$S/a8.wav" > "$S/alist.txt"
ffmpeg -y -f concat -safe 0 -i "$S/alist.txt" -c:a aac -b:a 192k "$S/audio.m4a" -loglevel error

OUTDIR=/home/zjy/code/lsy/vllm-omni-rankings/cosmos_edge_promo
ffmpeg -y -i "$S/video.mp4" -i "$S/audio.m4a" -c:v copy -c:a aac -shortest "$OUTDIR/cosmos_edge_promo_v5.mp4" -loglevel error
echo "DONE -> $OUTDIR/cosmos_edge_promo_v5.mp4"
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUTDIR/cosmos_edge_promo_v5.mp4"
