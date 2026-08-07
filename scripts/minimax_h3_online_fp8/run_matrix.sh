#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (( $# == 0 )); then
  cases=(
    bf16 dit_fp8
    te_mlp_l0_9 te_mlp_l0_24
    te_mlp_all te_mlp_o_all
    te_all_linear dit_te_mlp_all
    all_linear_fp8
  )
else
  cases=("$@")
fi

run_pair() {
  local left=$1
  local right=${2:-}
  local left_pid
  local right_pid=

  bash "$script_dir/run_case.sh" "$left" 0,1,2,3 8091 &
  left_pid=$!
  if [[ -n $right ]]; then
    bash "$script_dir/run_case.sh" "$right" 4,5,6,7 8093 &
    right_pid=$!
  fi

  local status=0
  wait "$left_pid" || status=$?
  if [[ -n $right_pid ]]; then
    wait "$right_pid" || status=$?
  fi
  return "$status"
}

for ((index = 0; index < ${#cases[@]}; index += 2)); do
  run_pair "${cases[index]}" "${cases[index + 1]:-}"
done
