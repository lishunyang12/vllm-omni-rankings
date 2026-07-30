#!/bin/bash
# Run Wan2.2 I2V dense/int8/skip/combo on four B200 GPUs in parallel, then launch SDPA on the
# first GPU that becomes free. Produces isolated logs/NPYs, LPIPS(all/first16), and MP4s.
set -euo pipefail
cd "$(dirname "$0")"

if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
  echo "Bash >= 5.1 is required for wait -n -p." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$HOME/cuda13}"
export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-$HOME/omni-env/bin/python}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPU_IDS <<<"$GPU_LIST"
if ((${#GPU_IDS[@]} != 4)); then
  echo "GPU_LIST must contain exactly four GPU IDs (default: '0 1 2 3')." >&2
  exit 1
fi

STEPS="${STEPS:-50}"
H="${H:-720}"
W="${W:-1280}"
FRAMES="${FRAMES:-81}"
THR="${THR:-0.5}"
U="${U:-0.94}"
IMG="${IMG:-bird_input_clean.png}"
IMG_URL="${IMG_URL:-https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png}"
PROMPT="${PROMPT:-CG animation style, a small blue bird takes off from a branch and lands on another branch}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-combo_parallel_i2v_out_${RUN_TAG}}"
mkdir -p "$OUT"

LOCK_FILE="${TMPDIR:-/tmp}/${USER}_wan22_combo_parallel_i2v.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another Wan2.2 parallel combo benchmark is already running on this node" >&2
  exit 1
fi

OMNI_ROOT="$("$PY" -c 'import pathlib, vllm_omni; print(pathlib.Path(vllm_omni.__file__).resolve().parents[1])')"
OMNI_HEAD="$(git -C "$OMNI_ROOT" rev-parse HEAD)"
SAGE_GRAPH_FIX="4197b9d565d2481b98b0ef59131c6b7ff1cb6269"
if ! git -C "$OMNI_ROOT" merge-base --is-ancestor "$SAGE_GRAPH_FIX" "$OMNI_HEAD"; then
  echo "installed vLLM-Omni source is missing the SAGE Dynamo graph-break fix" >&2
  echo "source=$OMNI_ROOT head=$OMNI_HEAD required=$SAGE_GRAPH_FIX" >&2
  exit 1
fi

RAW=https://raw.githubusercontent.com/lishunyang12/vllm-omni-rankings/main/scripts/wan22_sage_e2e
[ -f wan22_combo_i2v.py ] || curl -fsSL "$RAW/wan22_combo_i2v.py" -o wan22_combo_i2v.py
[ -f "$IMG" ] || curl -fL "$IMG_URL" -o "$IMG"
cp -f "$IMG" "$OUT/i2v_input.png"

echo "vllm-omni source=$OMNI_ROOT head=$OMNI_HEAD"
echo "GPUs=${GPU_IDS[*]} output=$OUT"
echo "image=$IMG prompt=$PROMPT"

declare -a ALL_PIDS=()
declare -a INITIAL_MODES=(dense int8 skip combo)
declare -A PID_BY_MODE=()
declare -A GPU_BY_PID=()
SDPA_PID=""

launch() {
  local label="$1"
  local gpu="$2"
  shift 2
  echo "launching $label on GPU $gpu -> $OUT/gen_${label}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" wan22_combo_i2v.py \
    "$@" \
    --image "$IMG" \
    --prompt "$PROMPT" \
    --steps "$STEPS" \
    --h "$H" \
    --w "$W" \
    --frames "$FRAMES" \
    --save "$OUT/${label}.npy" \
    >"$OUT/gen_${label}.log" 2>&1 &
  local pid=$!
  ALL_PIDS+=("$pid")
  PID_BY_MODE["$label"]="$pid"
  GPU_BY_PID["$pid"]="$gpu"
}

stop_children() {
  local pid
  for pid in "${ALL_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap 'stop_children; exit 130' INT TERM

launch dense "${GPU_IDS[0]}" --mode dense
launch int8 "${GPU_IDS[1]}" --mode int8
launch skip "${GPU_IDS[2]}" --mode skip --threshold "$THR" --until "$U"
launch combo "${GPU_IDS[3]}" --mode combo --threshold "$THR" --until "$U"

echo "waiting for the first GPU to become free..."
if wait -n -p FIRST_PID "${ALL_PIDS[@]}"; then
  rc=0
else
  rc=$?
fi
if ((rc != 0)); then
  echo "the first completed task failed (rc=$rc); inspect $OUT/gen_*.log" >&2
  stop_children
  wait || true
  exit "$rc"
fi

FIRST_MODE=""
for mode in "${INITIAL_MODES[@]}"; do
  if [[ "${PID_BY_MODE[$mode]}" == "$FIRST_PID" ]]; then
    FIRST_MODE="$mode"
    break
  fi
done
if [[ -z "$FIRST_MODE" ]]; then
  echo "could not map completed PID $FIRST_PID to a mode" >&2
  stop_children
  wait || true
  exit 1
fi

FREE_GPU="${GPU_BY_PID[$FIRST_PID]}"
echo "$FIRST_MODE finished first; launching SDPA on freed GPU $FREE_GPU"
launch sdpa "$FREE_GPU" --mode sdpa
SDPA_PID="${PID_BY_MODE[sdpa]}"

failed=0
for mode in "${INITIAL_MODES[@]}"; do
  pid="${PID_BY_MODE[$mode]}"
  if [[ "$pid" == "$FIRST_PID" ]]; then
    continue
  fi
  if wait "$pid"; then
    echo "$mode finished"
  else
    echo "$mode failed; inspect $OUT/gen_${mode}.log" >&2
    failed=1
  fi
done
if wait "$SDPA_PID"; then
  echo "sdpa finished"
else
  echo "sdpa failed; inspect $OUT/gen_sdpa.log" >&2
  failed=1
fi
trap - INT TERM
if ((failed)); then
  exit 1
fi

echo "########## RESULTS (vs dense) ##########"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" OUT="$OUT" STEPS="$STEPS" THR="$THR" U="$U" "$PY" - <<'PY'
import os
import re

import lpips
import numpy as np
import torch

out = os.environ["OUT"]
steps = int(os.environ["STEPS"])
threshold = os.environ["THR"]
until = os.environ["U"]


def load(path):
    array = np.load(path).squeeze()
    if array.dtype != np.uint8 and float(array.max()) <= 1.0 + 1e-3:
        array = array * 255.0
    return array.clip(0, 255).astype("uint8")


def to_tensor(array):
    return (torch.from_numpy(np.ascontiguousarray(array)).float() / 127.5 - 1.0).permute(0, 3, 1, 2)


def generation_time(label):
    text = open(f"{out}/gen_{label}.log").read()
    return float(re.search(r"generation\s*:\s*([\d.]+)", text).group(1))


device = "cuda" if torch.cuda.is_available() else "cpu"
loss = lpips.LPIPS(net="alex").to(device).eval()
dense = to_tensor(load(f"{out}/dense.npy"))
num_frames = dense.shape[0]
dense_time = generation_time("dense")


def lpips_mean(candidate, start, stop):
    with torch.no_grad():
        return float(
            np.mean(
                [
                    loss(dense[i : i + 1].to(device), candidate[i : i + 1].to(device)).item()
                    for i in range(start, stop)
                ]
            )
        )


rows = [
    ("dense (trtllm)", "dense"),
    ("int8 SAGE", "int8"),
    (f"skip @thr{threshold}/u{until}", "skip"),
    (f"int8+skip @thr{threshold}/u{until}", "combo"),
    ("SDPA", "sdpa"),
]
print("\n| config | s/step | vs dense | LPIPS all | LPIPS first16 |")
print("|---|---:|---:|---:|---:|")
for name, label in rows:
    elapsed = generation_time(label)
    if label == "dense":
        speedup = lpips_all = lpips_first16 = "—"
    else:
        candidate = to_tensor(load(f"{out}/{label}.npy"))
        speedup = f"{dense_time / elapsed:.3f}×"
        lpips_all = f"{lpips_mean(candidate, 0, num_frames):.4f}"
        lpips_first16 = f"{lpips_mean(candidate, 0, min(16, num_frames)):.4f}"
    print(f"| {name} | {elapsed / steps:.3f} | {speedup} | {lpips_all} | {lpips_first16} |")

try:
    import imageio.v2 as imageio

    for _, label in rows:
        array = load(f"{out}/{label}.npy")
        writer = imageio.get_writer(
            f"{out}/{label}.mp4",
            fps=16,
            codec="libx264",
            quality=6,
            macro_block_size=8,
            ffmpeg_params=["-crf", "26", "-pix_fmt", "yuv420p"],
        )
        for frame in array:
            writer.append_data(frame)
        writer.close()
        print("wrote", f"{out}/{label}.mp4")
except Exception as exc:
    print("VIDEO SKIPPED:", exc)
PY

echo "########## DONE -> $OUT/ ##########"
