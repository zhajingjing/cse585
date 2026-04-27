#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SERVING_SCRIPT="${SERVING_SCRIPT:-$ROOT_DIR/serving_system_video_teacache.py}"

DEFAULT_CKPT_DIR="/root/autodl-tmp/Wan2.1-T2V-1.3B"
CKPT_DIR="${1:-${CKPT_DIR:-$DEFAULT_CKPT_DIR}}"

if [[ -z "$CKPT_DIR" ]]; then
  echo "Usage: $0 /path/to/wan_checkpoint_dir"
  echo "Or set CKPT_DIR in the environment."
  exit 1
fi

# ── Fixed generation settings ────────────────────────────────────────────────
TASK="${TASK:-t2v-1.3B}"
SIZE="${SIZE:-832*480}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5.0}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
TEACACHE_THRESH="${TEACACHE_THRESH:-0.2}"
LOOP="${LOOP:-1}"
NUM_REQ="${NUM_REQ:-50}"
ENABLE_LOG="${ENABLE_LOG:-1}"

# ── Request arrival & prefill ────────────────────────────────────────────────
# 30 s interval: request 0 finishes at ~30 s and populates the cache,
# so request 1 (at 30 s) is the first that can get a cache hit.
REQUEST_INTERVAL_SECONDS="${REQUEST_INTERVAL_SECONDS:-30}"

# ── Experiment axes ──────────────────────────────────────────────────────────
# Cache sizes to sweep (number of distinct prompts that can be cached).
# WARMUP_SIZES is paired 1-to-1 with CACHE_SIZES: for a cache of size N,
# seed it with the corresponding number of warmup requests so the cache is
# meaningfully populated before the timed experiment begins.
CACHE_SIZES=(  10)
WARMUP_SIZES=( 10)

# Cache / serving policies:
#   nirvana_teacache  – Nirvana latent cache + TeaCache block skipping
#   teacache_only     – TeaCache only, no Nirvana latent cache
#   none              – plain Wan, no caching at all (baseline)
METHODS=(
  "lcbfu:"
  "lru:--eviction_policy lru"
  "fifo:--eviction_policy fifo"
)

# ── Workloads ────────────────────────────────────────────────────────────────
WORKLOADS=(
  "moderately_similar_50:eval/teacache/vbench/workload_moderately_similar_50_shuffled.json"
  # "realistic_50:eval/teacache/vbench/workload_realistic_50.json"
  # "highly_similar_20:eval/teacache/vbench/workload_highly_similar_20.json"
)

# ── Output ───────────────────────────────────────────────────────────────────
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/serving_runs/$RUN_STAMP}"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

mkdir -p "$OUT_ROOT"
echo "workload,method,cache_size,\
wall_total_s,\
latency_min_s,latency_max_s,latency_avg_s,latency_n,\
hit_processing_avg_s,hit_processing_n,\
miss_processing_avg_s,miss_processing_n,\
cache_hits,cache_total,cache_hit_rate_pct,\
vector_search_avg_ms,\
run_dir,console_log" > "$SUMMARY_CSV"

# ── Helpers ──────────────────────────────────────────────────────────────────
parse_summary_to_csv_row() {
  local workload="$1" method="$2" cache_size="$3" run_dir="$4" console_log="$5"

  local wall="" lat_min="" lat_max="" lat_avg="" lat_n=""
  local hit_avg="" hit_n="" miss_avg="" miss_n=""
  local cache_hits="" cache_total="" cache_pct=""
  local vsearch_avg=""

  wall="$(grep '\[Total wall time\]' "$console_log" | sed -n 's/.*serving only: \([0-9.]*\)s.*/\1/p' || true)"

  local lat_line hit_line miss_line cache_line vsearch_line
  lat_line="$(grep     '\[Per-request latency\]'  "$console_log" | tail -n1 || true)"
  hit_line="$(grep     '\[Hit processing time\]'  "$console_log" | tail -n1 || true)"
  miss_line="$(grep    '\[Miss processing time\]' "$console_log" | tail -n1 || true)"
  cache_line="$(grep   '\[Cache hit rate\]'        "$console_log" | tail -n1 || true)"
  vsearch_line="$(grep '\[Vector search time\]'   "$console_log" | tail -n1 || true)"

  [[ -n "$lat_line" ]] && {
    lat_min="$(echo "$lat_line"  | sed -n 's/.*min=\([0-9.]*\)s.*/\1/p')"
    lat_max="$(echo "$lat_line"  | sed -n 's/.*max=\([0-9.]*\)s.*/\1/p')"
    lat_avg="$(echo "$lat_line"  | sed -n 's/.*avg=\([0-9.]*\)s.*/\1/p')"
    lat_n="$(echo   "$lat_line"  | sed -n 's/.*(n=\([0-9]*\)).*/\1/p')"
  }
  [[ -n "$hit_line" ]] && {
    hit_avg="$(echo "$hit_line"  | sed -n 's/.*avg=\([0-9.]*\)s.*/\1/p')"
    hit_n="$(echo   "$hit_line"  | sed -n 's/.*(n=\([0-9]*\)).*/\1/p')"
  }
  [[ -n "$miss_line" ]] && {
    miss_avg="$(echo "$miss_line" | sed -n 's/.*avg=\([0-9.]*\)s.*/\1/p')"
    miss_n="$(echo   "$miss_line" | sed -n 's/.*(n=\([0-9]*\)).*/\1/p')"
  }
  if [[ "$cache_line" == *"N/A"* ]]; then
    cache_hits="0"; cache_total="0"; cache_pct="0"
  elif [[ -n "$cache_line" ]]; then
    cache_hits="$(echo  "$cache_line" | sed -n 's/.*] \([0-9]*\)\/\([0-9]*\) = \([0-9.]*\)%.*/\1/p')"
    cache_total="$(echo "$cache_line" | sed -n 's/.*] \([0-9]*\)\/\([0-9]*\) = \([0-9.]*\)%.*/\2/p')"
    cache_pct="$(echo   "$cache_line" | sed -n 's/.*] \([0-9]*\)\/\([0-9]*\) = \([0-9.]*\)%.*/\3/p')"
  fi
  [[ -n "$vsearch_line" ]] && \
    vsearch_avg="$(echo "$vsearch_line" | sed -n 's/.*avg=\([0-9.]*\)ms.*/\1/p')"

  echo "$workload,$method,$cache_size,\
$wall,\
$lat_min,$lat_max,$lat_avg,$lat_n,\
$hit_avg,$hit_n,\
$miss_avg,$miss_n,\
$cache_hits,$cache_total,$cache_pct,\
$vsearch_avg,\
$run_dir,$console_log" >> "$SUMMARY_CSV"
}

# ── Main loop ────────────────────────────────────────────────────────────────
echo "Output root: $OUT_ROOT"
echo "Checkpoint:  $CKPT_DIR"
echo "Sweep: ${#CACHE_SIZES[@]} cache sizes × ${#METHODS[@]} methods × ${#WORKLOADS[@]} workloads  [warmup sizes: ${WARMUP_SIZES[*]}]"

for workload_spec in "${WORKLOADS[@]}"; do
  workload_name="${workload_spec%%:*}"
  workload_path="${workload_spec#*:}"
  abs_workload_path="$ROOT_DIR/$workload_path"

  if [[ ! -f "$abs_workload_path" ]]; then
    echo "Missing workload file: $abs_workload_path" >&2
    exit 1
  fi

  for cache_idx in "${!CACHE_SIZES[@]}"; do
    cache_size="${CACHE_SIZES[$cache_idx]}"
    warmup_requests="${WARMUP_SIZES[$cache_idx]}"

    for method_spec in "${METHODS[@]}"; do
      method_name="${method_spec%%:*}"
      method_flags="${method_spec#*:}"

      run_dir="$OUT_ROOT/$workload_name/cache${cache_size}/$method_name"
      video_dir="$run_dir/videos"
      mkdir -p "$video_dir"

      console_log="$run_dir/console.log"
      throughput_csv="$run_dir/request_throughput.csv"
      command_txt="$run_dir/command.txt"

      cmd=(
        "$PYTHON_BIN" "$SERVING_SCRIPT"
        "--ckpt_dir"               "$CKPT_DIR"
        "--task"                   "$TASK"
        "--prompt_list"            "$abs_workload_path"
        "--video_directory"        "$video_dir"
        "--cache_size"             "$cache_size"
        "--size"                   "$SIZE"
        "--num_frames"             "$NUM_FRAMES"
        "--sample_steps"           "$SAMPLE_STEPS"
        "--sample_solver"          "$SAMPLE_SOLVER"
        "--sample_shift"           "$SAMPLE_SHIFT"
        "--guide_scale"            "$GUIDE_SCALE"
        "--teacache_thresh"        "$TEACACHE_THRESH"
        "--loop"                   "$LOOP"
        "--log_file"               "$throughput_csv"
        "--request_interval_seconds" "$REQUEST_INTERVAL_SECONDS"
        "--warmup_requests"        "$warmup_requests"
      )

      [[ -n "$NUM_REQ" ]]       && cmd+=("--num_req" "$NUM_REQ")
      [[ "$ENABLE_LOG" == "1" ]] && cmd+=("--log")

      if [[ -n "$method_flags" ]]; then
        # shellcheck disable=SC2206
        extra_flags=($method_flags)
        cmd+=("${extra_flags[@]}")
      fi

      printf '%q ' "${cmd[@]}" > "$command_txt"
      echo >> "$command_txt"

      echo
      echo "=== workload=$workload_name  cache_size=$cache_size  warmup=$warmup_requests  method=$method_name ==="
      echo "Run dir: $run_dir"

      "${cmd[@]}" 2>&1 | tee "$console_log"

      parse_summary_to_csv_row \
        "$workload_name" "$method_name" "$cache_size" "$run_dir" "$console_log"
    done
  done
done

echo
echo "Finished all runs."
echo "Summary: $SUMMARY_CSV"
