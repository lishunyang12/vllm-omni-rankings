#!/usr/bin/env bash
set -euo pipefail

case_name=${1:?usage: run_case.sh CASE GPU_LIST PORT}
gpu_list=${2:?usage: run_case.sh CASE GPU_LIST PORT}
port=${3:?usage: run_case.sh CASE GPU_LIST PORT}

model_root=${MODEL_ROOT:-/mnt/minecraft/model/MiniMax-H3}
omni_root=${OMNI_ROOT:-/home/zjy/code/lsy/worktree/minimax-h3-b300}
output_root=${OUTPUT_ROOT:-/home/zjy/code/lsy/bench_outputs/minimax_h3_online_fp8_ablation}
vllm_bin=${VLLM_BIN:-$omni_root/.venv/bin/vllm}
i2va_image=${I2VA_IMAGE:-/tmp/minimax_h3_i2va_first_frame.png}
ref_video=${REF_VIDEO:-$model_root/ref2va_official_inputs/reference_video.mp4}
bench_tasks=,${BENCH_TASKS:-t2va,i2va,ref2va},
warmup_steps=${WARMUP_STEPS:-2}

wants_task() {
  [[ $bench_tasks == *,$1,* ]]
}

if [[ ! -f $i2va_image ]]; then
  echo "I2VA_IMAGE does not exist: $i2va_image" >&2
  exit 2
fi
if [[ ! -f $ref_video ]]; then
  echo "REF_VIDEO does not exist: $ref_video" >&2
  exit 2
fi

build_quant_config() {
  case "$case_name" in
    bf16)
      printf ''
      ;;
    dit_fp8)
      jq -cn '{transformer:{method:"fp8"},text_encoder:null}'
      ;;
    te_mlp_l0_9)
      jq -cn 'reduce range(0;10) as $i ({transformer:null,text_encoder:null}; .["text_encoder.text_model.layers."+($i|tostring)+".mlp"]={method:"fp8"})'
      ;;
    te_mlp_l0_24)
      jq -cn 'reduce range(0;25) as $i ({transformer:null,text_encoder:null}; .["text_encoder.text_model.layers."+($i|tostring)+".mlp"]={method:"fp8"})'
      ;;
    te_mlp_all)
      jq -cn 'reduce range(0;50) as $i ({transformer:null,text_encoder:null}; .["text_encoder.text_model.layers."+($i|tostring)+".mlp"]={method:"fp8"})'
      ;;
    te_mlp_o_all)
      jq -cn 'reduce range(0;50) as $i ({transformer:null,text_encoder:null}; .["text_encoder.text_model.layers."+($i|tostring)+".mlp"]={method:"fp8"} | .["text_encoder.text_model.layers."+($i|tostring)+".self_attn.o_proj"]={method:"fp8"})'
      ;;
    te_all_linear)
      jq -cn '{transformer:null,text_encoder:{method:"fp8"}}'
      ;;
    dit_te_mlp_all)
      jq -cn 'reduce range(0;50) as $i ({transformer:{method:"fp8"},text_encoder:null}; .["text_encoder.text_model.layers."+($i|tostring)+".mlp"]={method:"fp8"})'
      ;;
    all_linear_fp8)
      printf 'global_fp8'
      ;;
    *)
      echo "unknown case: $case_name" >&2
      exit 2
      ;;
  esac
}

quant_config=$(build_quant_config)
quant_args=()
if [[ $quant_config == global_fp8 ]]; then
  quant_args=(--quantization fp8)
elif [[ -n $quant_config ]]; then
  quant_args=(--diffusion-quantization-config "$quant_config")
fi

server_pid=
stop_server() {
  if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid"
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
trap stop_server EXIT

wait_for_server() {
  local log_file=$1
  for _ in $(seq 1 120); do
    if curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 100 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "server did not become ready: $log_file" >&2
  tail -n 100 "$log_file" >&2
  return 1
}

start_server() {
  local partition=$1
  local log_file=$2
  CUDA_VISIBLE_DEVICES=$gpu_list \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  "$vllm_bin" serve "$model_root/$partition" \
    --omni \
    --host 127.0.0.1 \
    --port "$port" \
    --trust-remote-code \
    --num-gpus 4 \
    --usp 4 \
    --ring 1 \
    --text-encoder-tp-size 4 \
    --vae-patch-parallel-size 4 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --diffusion-attention-backend CUDNN_ATTN \
    "${quant_args[@]}" >"$log_file" 2>&1 &
  server_pid=$!
  wait_for_server "$log_file"
}

prompt_from_script() {
  local script=$1
  sed -n '/^{/,/^}$/p' "$script" | jq -r '.prompt'
}

record_gpu() {
  local path=$1
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits --loop-ms=500 >"$path" &
}

validate_video() {
  local video=$1
  ffprobe -v error \
    -show_entries format=duration,size \
    -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels \
    -of json "$video" >"${video%.mp4}.media.json"
}

run_t2va() {
  local destination=$1
  local steps=${2:-50}
  local prompt
  prompt=$(prompt_from_script "$model_root/scripts/readme/reproducible-768p-t2va-request.sh")
  curl --fail-with-body --silent --show-error --max-time 1800 \
    --request POST "http://127.0.0.1:$port/v1/videos/sync" \
    --form-string "prompt=$prompt" \
    -F fps=24 -F "num_inference_steps=$steps" -F flow_shift=12 -F seed=0 \
    -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
    --output "$destination"
}

run_i2va() {
  local destination=$1
  local steps=${2:-50}
  local prompt
  prompt=$(prompt_from_script "$model_root/scripts/readme/reproducible-768p-fl2va-request.sh")
  curl --fail-with-body --silent --show-error --max-time 1800 \
    --request POST "http://127.0.0.1:$port/v1/videos/sync" \
    --form-string "prompt=$prompt" \
    -F fps=24 -F "num_inference_steps=$steps" -F flow_shift=12 -F seed=0 \
    -F 'extra_params={"task":"fl2va","duration":5.0,"audio_flow_shift":3.0}' \
    -F "input_reference=@$i2va_image;type=image/png" \
    --output "$destination"
}

run_ref2va() {
  local destination=$1
  local steps=${2:-50}
  local prompt
  prompt=$(prompt_from_script "$model_root/scripts/readme/reproducible-768p-ref2va-request.sh")
  curl --fail-with-body --silent --show-error --max-time 1800 \
    --request POST "http://127.0.0.1:$port/v1/videos/sync" \
    --form-string "prompt=$prompt" \
    -F fps=24 -F "num_inference_steps=$steps" -F flow_shift=12 -F seed=0 \
    -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
    -F "input_references=@$ref_video;type=video/mp4" \
    --output "$destination"
}

run_measured() {
  local task=$1
  local runner=$2
  local destination=$3
  mkdir -p "$(dirname "$destination")"
  local monitor_pid
  local started_ns
  local finished_ns
  record_gpu "${destination%.mp4}.gpu.csv"
  monitor_pid=$!
  started_ns=$(date +%s%N)
  set +e
  "$runner" "$destination"
  local status=$?
  set -e
  finished_ns=$(date +%s%N)
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  jq -n \
    --arg case "$case_name" \
    --arg task "$task" \
    --arg gpu_list "$gpu_list" \
    --argjson started_ns "$started_ns" \
    --argjson finished_ns "$finished_ns" \
    --argjson exit_status "$status" \
    '{case:$case,task:$task,gpus:$gpu_list,started_ns:$started_ns,finished_ns:$finished_ns,wall_time_s:(($finished_ns-$started_ns)/1000000000),exit_status:$exit_status}' \
    >"${destination%.mp4}.time.json"
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi
  validate_video "$destination"
  printf '%s %s %s\n' "$case_name" "$task" "$destination"
  return "$status"
}

case_root=$output_root/cases/$case_name
mkdir -p "$case_root/fl2va" "$case_root/ref2va"
if [[ $quant_config == global_fp8 ]]; then
  printf '"fp8"\n' >"$case_root/quantization_config.json"
elif [[ -n $quant_config ]]; then
  printf '%s\n' "$quant_config" >"$case_root/quantization_config.json"
else
  printf 'null\n' >"$case_root/quantization_config.json"
fi
printf '%s\n' "$gpu_list" >"$case_root/gpus.txt"

if wants_task t2va || wants_task i2va; then
  start_server FL2VA "$case_root/fl2va/server.log"
  if wants_task t2va; then
    run_t2va "/tmp/minimax_h3_${case_name}_t2va_warmup.mp4" "$warmup_steps"
    run_measured t2va run_t2va "$case_root/t2va.mp4"
  else
    run_i2va "/tmp/minimax_h3_${case_name}_i2va_warmup.mp4" "$warmup_steps"
  fi
  if wants_task i2va; then
    run_measured i2va run_i2va "$case_root/i2va.mp4"
  fi
  stop_server
fi

if wants_task ref2va; then
  start_server Ref2VA "$case_root/ref2va/server.log"
  run_ref2va "/tmp/minimax_h3_${case_name}_ref2va_warmup.mp4" "$warmup_steps"
  run_measured ref2va run_ref2va "$case_root/ref2va.mp4"
  stop_server
fi

if [[ -s $case_root/t2va.mp4 && -s $case_root/i2va.mp4 && -s $case_root/ref2va.mp4 ]]; then
  printf 'complete\n' >"$case_root/status.txt"
else
  printf 'partial:%s\n' "${BENCH_TASKS:-t2va,i2va,ref2va}" >"$case_root/status.txt"
fi
