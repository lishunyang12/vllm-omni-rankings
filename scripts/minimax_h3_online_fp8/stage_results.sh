#!/usr/bin/env bash
set -euo pipefail

output_root=${1:?usage: stage_results.sh OUTPUT_ROOT}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cases=(
  bf16 dit_fp8 te_mlp_l0_9 te_mlp_l0_24 te_mlp_all te_mlp_o_all
  te_all_linear dit_te_mlp_all all_linear_fp8
)
tasks=(t2va i2va ref2va)

for case_name in "${cases[@]}"; do
  source_dir=$output_root/cases/$case_name
  video_dir=$script_dir/videos/$case_name
  measurement_dir=$script_dir/measurements/$case_name
  mkdir -p "$video_dir" "$measurement_dir"
  for task in "${tasks[@]}"; do
    for suffix in mp4 time.json media.json gpu.csv; do
      source_file=$source_dir/$task.$suffix
      if [[ ! -s $source_file ]]; then
        echo "missing benchmark artifact: $source_file" >&2
        exit 1
      fi
      if [[ $suffix == mp4 ]]; then
        cp "$source_file" "$video_dir/$task.mp4"
      else
        cp "$source_file" "$measurement_dir/$task.$suffix"
      fi
    done
  done
  cp "$source_dir/quantization_config.json" "$measurement_dir/quantization_config.json"
done

mkdir -p "$script_dir/results"
cp "$output_root/aggregate/results.csv" "$script_dir/results/results.csv"
cp "$output_root/aggregate/results.json" "$script_dir/results/results.json"
cp "$output_root/aggregate/summary.json" "$script_dir/results/summary.json"
cp "$output_root/aggregate/RESULTS.md" "$script_dir/RESULTS.md"

dlo_source=$output_root/dlo_smoke
dlo_dir=$script_dir/dlo_smoke
mkdir -p "$dlo_dir"
for filename in t2va.mp4 time.json media.json gpu.csv; do
  source_file=$dlo_source/$filename
  if [[ ! -s $source_file ]]; then
    echo "missing DLO smoke artifact: $source_file" >&2
    exit 1
  fi
  cp "$source_file" "$dlo_dir/$filename"
done

(
  cd "$script_dir"
  find videos dlo_smoke -type f -name '*.mp4' -print0 \
    | sort -z | xargs -0 sha256sum >SHA256SUMS
)
