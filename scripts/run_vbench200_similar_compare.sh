#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SERVING_SCRIPT="${SERVING_SCRIPT:-$SCRIPT_DIR/serving/serving_system_video_teacache.py}"

DEFAULT_CKPT_DIR="$PROJECT_ROOT/pretrained/Wan2.1-T2V-1.3B"
CKPT_DIR="${1:-${CKPT_DIR:-$DEFAULT_CKPT_DIR}}"

TASK="${TASK:-t2v-1.3B}"
SIZE="${SIZE:-832*480}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5.0}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
TEACACHE_THRESH="${TEACACHE_THRESH:-0.2}"
CACHE_SIZE="${CACHE_SIZE:-1000}"
LOOP="${LOOP:-1}"
NUM_REQ="${NUM_REQ:-200}"
REQUEST_INTERVAL_SECONDS="${REQUEST_INTERVAL_SECONDS:-}"
ENABLE_LOG="${ENABLE_LOG:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

WORKLOAD_PATH="${WORKLOAD_PATH:-$PROJECT_ROOT/eval/teacache/vbench/VBench_200_similar.json}"
if [[ ! -f "$WORKLOAD_PATH" ]]; then
  echo "Missing workload file: $WORKLOAD_PATH" >&2
  exit 1
fi

RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/vbench200_similar_runs/$RUN_STAMP}"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

mkdir -p "$OUT_ROOT"
echo "method,run_dir,console_log,throughput_csv,total_wall_time_s,latency_min_s,latency_max_s,latency_avg_s,latency_n,processing_min_s,processing_max_s,processing_avg_s,processing_n,hit_proc_min_s,hit_proc_max_s,hit_proc_avg_s,hit_proc_n,miss_proc_min_s,miss_proc_max_s,miss_proc_avg_s,miss_proc_n,hit_gen_min_s,hit_gen_max_s,hit_gen_avg_s,hit_gen_n,miss_gen_min_s,miss_gen_max_s,miss_gen_avg_s,miss_gen_n,vsearch_min_ms,vsearch_max_ms,vsearch_avg_ms,vsearch_n,retrieval_min_ms,retrieval_max_ms,retrieval_avg_ms,retrieval_n,cache_hits,cache_total,cache_hit_rate_pct" > "$SUMMARY_CSV"

extract_metric() {
  local pattern="$1"
  local file="$2"
  grep -oE "$pattern" "$file" | head -n1 || true
}

append_summary_row() {
  local method="$1"
  local run_dir="$2"
  local console_log="$3"
  local throughput_csv="$4"

  local wall latency_line proc_line hit_proc_line miss_proc_line
  local hit_gen_line miss_gen_line vsearch_line retrieval_line cache_line
  wall="$(extract_metric '\[Total wall time\] [0-9.]+s' "$console_log" | grep -oE '[0-9.]+')"
  latency_line="$(grep '\[Per-request latency\]' "$console_log" | tail -n1 || true)"
  proc_line="$(grep '\[Pure processing time\]' "$console_log" | tail -n1 || true)"
  hit_proc_line="$(grep '\[Hit processing time\]' "$console_log" | tail -n1 || true)"
  miss_proc_line="$(grep '\[Miss processing time\]' "$console_log" | tail -n1 || true)"
  hit_gen_line="$(grep '\[Hit generation time\]' "$console_log" | tail -n1 || true)"
  miss_gen_line="$(grep '\[Miss generation time\]' "$console_log" | tail -n1 || true)"
  vsearch_line="$(grep '\[Vector search time\]' "$console_log" | tail -n1 || true)"
  retrieval_line="$(grep '\[Cache retrieval time\]' "$console_log" | tail -n1 || true)"
  cache_line="$(grep '\[Cache hit rate\]' "$console_log" | tail -n1 || true)"

  local lat_min="" lat_max="" lat_avg="" lat_n=""
  local proc_min="" proc_max="" proc_avg="" proc_n=""
  local hit_proc_min="" hit_proc_max="" hit_proc_avg="" hit_proc_n=""
  local miss_proc_min="" miss_proc_max="" miss_proc_avg="" miss_proc_n=""
  local hit_gen_min="" hit_gen_max="" hit_gen_avg="" hit_gen_n=""
  local miss_gen_min="" miss_gen_max="" miss_gen_avg="" miss_gen_n=""
  local vsearch_min="" vsearch_max="" vsearch_avg="" vsearch_n=""
  local retrieval_min="" retrieval_max="" retrieval_avg="" retrieval_n=""
  local cache_hits="" cache_total="" cache_pct=""

  _parse_stats_s() {
    local line="$1"
    echo "$(echo "$line" | sed -n 's/.*min=\([0-9.]*\)s.*/\1/p')" \
         "$(echo "$line" | sed -n 's/.*max=\([0-9.]*\)s.*/\1/p')" \
         "$(echo "$line" | sed -n 's/.*avg=\([0-9.]*\)s.*/\1/p')" \
         "$(echo "$line" | sed -n 's/.*(n=\([0-9]*\)).*/\1/p')"
  }

  _parse_stats_ms() {
    local line="$1"
    echo "$(echo "$line" | sed -n 's/.*min=\([0-9.]*\)ms.*/\1/p')" \
         "$(echo "$line" | sed -n 's/.*max=\([0-9.]*\)ms.*/\1/p')" \
         "$(echo "$line" | sed -n 's/.*avg=\([0-9.]*\)ms.*/\1/p')" \
         "$(echo "$line" | sed -n 's/.*(n=\([0-9]*\)).*/\1/p')"
  }

  if [[ -n "$latency_line" ]]; then
    read -r lat_min lat_max lat_avg lat_n <<< "$(_parse_stats_s "$latency_line")"
  fi
  if [[ -n "$proc_line" ]]; then
    read -r proc_min proc_max proc_avg proc_n <<< "$(_parse_stats_s "$proc_line")"
  fi
  if [[ -n "$hit_proc_line" ]]; then
    read -r hit_proc_min hit_proc_max hit_proc_avg hit_proc_n <<< "$(_parse_stats_s "$hit_proc_line")"
  fi
  if [[ -n "$miss_proc_line" ]]; then
    read -r miss_proc_min miss_proc_max miss_proc_avg miss_proc_n <<< "$(_parse_stats_s "$miss_proc_line")"
  fi
  if [[ -n "$hit_gen_line" ]]; then
    read -r hit_gen_min hit_gen_max hit_gen_avg hit_gen_n <<< "$(_parse_stats_s "$hit_gen_line")"
  fi
  if [[ -n "$miss_gen_line" ]]; then
    read -r miss_gen_min miss_gen_max miss_gen_avg miss_gen_n <<< "$(_parse_stats_s "$miss_gen_line")"
  fi
  if [[ -n "$vsearch_line" ]]; then
    read -r vsearch_min vsearch_max vsearch_avg vsearch_n <<< "$(_parse_stats_ms "$vsearch_line")"
  fi
  if [[ -n "$retrieval_line" ]]; then
    read -r retrieval_min retrieval_max retrieval_avg retrieval_n <<< "$(_parse_stats_ms "$retrieval_line")"
  fi

  if [[ "$cache_line" == *"N/A"* ]]; then
    cache_hits="0"; cache_total="0"; cache_pct="0"
  elif [[ -n "$cache_line" ]]; then
    cache_hits="$(echo "$cache_line" | sed -n 's/.*] \([0-9]*\)\/\([0-9]*\) = \([0-9.]*\)%.*/\1/p')"
    cache_total="$(echo "$cache_line" | sed -n 's/.*] \([0-9]*\)\/\([0-9]*\) = \([0-9.]*\)%.*/\2/p')"
    cache_pct="$(echo "$cache_line" | sed -n 's/.*] \([0-9]*\)\/\([0-9]*\) = \([0-9.]*\)%.*/\3/p')"
  fi

  echo "$method,$run_dir,$console_log,$throughput_csv,$wall,$lat_min,$lat_max,$lat_avg,$lat_n,$proc_min,$proc_max,$proc_avg,$proc_n,$hit_proc_min,$hit_proc_max,$hit_proc_avg,$hit_proc_n,$miss_proc_min,$miss_proc_max,$miss_proc_avg,$miss_proc_n,$hit_gen_min,$hit_gen_max,$hit_gen_avg,$hit_gen_n,$miss_gen_min,$miss_gen_max,$miss_gen_avg,$miss_gen_n,$vsearch_min,$vsearch_max,$vsearch_avg,$vsearch_n,$retrieval_min,$retrieval_max,$retrieval_avg,$retrieval_n,$cache_hits,$cache_total,$cache_pct" >> "$SUMMARY_CSV"
}

run_method() {
  local method="$1"
  local extra_method_flag="$2"

  local run_dir="$OUT_ROOT/$method"
  local video_dir="$run_dir/videos"
  local console_log="$run_dir/console.log"
  local throughput_csv="$run_dir/request_throughput.csv"
  local command_txt="$run_dir/command.txt"
  mkdir -p "$video_dir"

  local -a cmd=(
    "$PYTHON_BIN" "$SERVING_SCRIPT"
    "--ckpt_dir" "$CKPT_DIR"
    "--task" "$TASK"
    "--prompt_list" "$WORKLOAD_PATH"
    "--video_directory" "$video_dir"
    "--cache_size" "$CACHE_SIZE"
    "--size" "$SIZE"
    "--num_frames" "$NUM_FRAMES"
    "--sample_steps" "$SAMPLE_STEPS"
    "--sample_solver" "$SAMPLE_SOLVER"
    "--sample_shift" "$SAMPLE_SHIFT"
    "--guide_scale" "$GUIDE_SCALE"
    "--teacache_thresh" "$TEACACHE_THRESH"
    "--loop" "$LOOP"
    "--num_req" "$NUM_REQ"
    "--log_file" "$throughput_csv"
  )

  if [[ -n "$REQUEST_INTERVAL_SECONDS" ]]; then
    cmd+=("--request_interval_seconds" "$REQUEST_INTERVAL_SECONDS")
  fi
  if [[ "$ENABLE_LOG" == "1" ]]; then
    cmd+=("--log")
  fi
  if [[ -n "$extra_method_flag" ]]; then
    cmd+=("$extra_method_flag")
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_args_array=($EXTRA_ARGS)
    cmd+=("${extra_args_array[@]}")
  fi

  printf '%q ' "${cmd[@]}" > "$command_txt"
  echo >> "$command_txt"

  echo
  echo "=== Running method=$method ==="
  echo "Workload: $WORKLOAD_PATH"
  echo "Run dir:  $run_dir"

  "${cmd[@]}" 2>&1 | tee "$console_log"
  append_summary_row "$method" "$run_dir" "$console_log" "$throughput_csv"
}

echo "Output root: $OUT_ROOT"
echo "Checkpoint:  $CKPT_DIR"
echo "Workload:    $WORKLOAD_PATH"

run_method "nirvana_teacache" ""
run_method "teacache_only" "--no_nirvana"

echo
echo "Finished runs."
echo "Summary written to: $SUMMARY_CSV"
