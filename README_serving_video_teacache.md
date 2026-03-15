# TeaCache Video Serving (Nirvana-Style Cross-Request Latent Reuse)

This document describes how to run **TeaCache video pipeline + Nirvana-style cross-request latent cache**: CLIP similarity + FAISS to find similar prompts, reuse or write video latents at steps 5/10/15, with KMinHeapCache eviction and multi-GPU workers.

---

## Environment and Dependencies

- **Python**: 3.8+ recommended
- **CUDA**: Match your PyTorch version
- Install project dependencies (see `requirements.txt`) and additionally:
  - `faiss-cpu` or `faiss-gpu` (vector search)
  - `pandas` (for request scheduling)

```bash
# If not already installed
pip install faiss-cpu pandas
# Or FAISS with GPU
pip install faiss-gpu pandas
```

---

## How to Run

**Run from the project root** so that `videosys`, `eval.teacache`, `serving_system_N`, etc. are importable:

```bash
cd /path/to/TeaCache
python serving_system_video_teacache.py [options]
```

---

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_req` | int | None | Max prompts to process (default: all when `--prompt_list` given, else 50). |
| `--cache_size` | int | 1000 | Max number of cached *requests* (each stores 3 steps 5/10/15; total cache entries = cache_size × 3). |
| `--video_directory` | str | `./video_outputs_teacache` | Directory to save generated videos (same role as opensora `output_dir`). |
| `--prompt_list` | str | None | JSON path; uses same `read_prompt_list` as opensora (`prompt_en`). |
| `--resolution` | str | 480p | Output resolution. |
| `--aspect_ratio` | str | 9:16 | Aspect ratio (e.g. 16:9, 1:1). |
| `--num_frames` | int | 51 | Number of video frames. |
| `--teacache_thresh` | float | 0.2 | TeaCache `rel_l1_thresh` (same as opensora teacache_fast). |
| `--loop` | int | 5 | Videos per prompt (same as opensora eval); only the first uses Nirvana cache. |
| `--eval_mode` | flag | False | No request timing: submit all prompts at once, run as fast as possible (like opensora eval). |
| `--no_nirvana` | flag | False | Disable cache: every request does full 30-step generation (for A/B time comparison). |

---

## Run Examples

### 1. Same flow as opensora eval (VBench + Nirvana)

```bash
python serving_system_video_teacache.py \
  --prompt_list eval/teacache/vbench/VBench_full_info.json \
  --video_directory ./samples/opensora_teacache_nirvana \
  --eval_mode
```

- Same prompt list as opensora (`read_prompt_list` → `prompt_en`), loop=5, output `{prompt}-{l}.mp4`, no request timing (`--eval_mode`). Nirvana cache is the only addition.

### 1b. Quick run (default prompts, 50 requests)

```bash
python serving_system_video_teacache.py --num_req 50 --video_directory ./video_outputs_teacache
```

- Uses 3 built-in prompts; 50 requests, 5 videos each (default loop=5). Naming: `{prompt}-0.mp4` … `{prompt}-4.mp4`.

### 2. VBench, first 20 prompts, 500 cache entries

```bash
python serving_system_video_teacache.py \
  --num_req 20 \
  --prompt_list eval/teacache/vbench/VBench_full_info.json \
  --cache_size 500 \
  --video_directory ./vbench_teacache_out
```

- Same `read_prompt_list` as opensora; runs first 20 prompts, 5 videos each; cache holds up to 500 requests’ latents.

### 3. Small cache (faster)

```bash
python serving_system_video_teacache.py \
  --num_req 10 \
  --cache_size 200 \
  --video_directory ./fast_out
```

- Fewer requests and smaller cache; good for debugging or latency tests.

### 4. Custom resolution and frame count

```bash
python serving_system_video_teacache.py \
  --num_req 30 \
  --resolution 360p \
  --aspect_ratio 16:9 \
  --num_frames 33 \
  --video_directory ./custom_out
```

- All requests use the same resolution, aspect ratio, and frame count (consistent with the cache policy).

---

## Behavior Overview

- **Scheduler process**: Dispatches requests by timestamp; encodes each prompt with CLIP and uses FAISS to find the nearest cached prompt; if similarity > 0.65, selects a step tier (0.95→step 15, 0.85→step 10, 0.65→step 5), retrieves the corresponding latent from KMinHeapCache and sends it with the request to a worker; otherwise marks as cache miss.
- **Worker process** (one per GPU): Loads Open-Sora + TeaCache (`teacache_forward`).  
  - **Miss**: Runs full 30-step generation, collects latents at steps 5/10/15, and pushes them with the CLIP embedding to `new_cache_queue`.  
  - **Hit**: Uses `cache_latent` and `cache_start_step` (5/10/15) from the request to continue denoising from that step to the end.
- **Cache**: KMinHeapCache with LCBFU eviction; k_values are fixed at `[5, 10, 15]`.

---

## Output and Logs

- **Videos**: Saved under `--video_directory`; filenames include prompt and timestamp.
- **Throughput log**: Written to `request_throughput_video_teacache.csv` in the current directory (timestamp, request_rate, throughput).
- **Console**: Prints latency stats (min/max/avg in seconds) when finished.

---

## FAQ

1. **ModuleNotFoundError: No module named 'eval'**  
   Run the script from the **project root** so the `eval` package is on the Python path.

2. **CUDA out of memory**  
   Reduce `--num_frames` or `--resolution`, or use fewer workers (currently one per GPU).

3. **FAISS-related errors**  
   Ensure `faiss-cpu` or `faiss-gpu` is installed and compatible with your Python/CUDA.

4. **Comparing with vs without Nirvana (time)**

   Run the same prompts twice and compare **Total wall time** and **Per-request latency (avg)**:

   ```bash
   # With Nirvana (cache enabled)
   python serving_system_video_teacache.py --prompt_list eval/teacache/vbench/VBench_full_info.json --num_req 20 --video_directory ./out_with_nirvana --eval_mode

   # Without Nirvana (every request full 30 steps)
   python serving_system_video_teacache.py --prompt_list eval/teacache/vbench/VBench_full_info.json --num_req 20 --video_directory ./out_no_nirvana --eval_mode --no_nirvana
   ```

   Use different `--video_directory` so outputs don’t overwrite. With Nirvana, similar prompts can hit cache and run fewer steps (faster).

5. **Changing k_values**  
   The implementation uses [5, 10, 15] by default. To use other steps, edit `K_VALUES_VIDEO` in `serving_system_video_teacache.py` and the corresponding `collect_latents_at_steps` / `cache_start_step` in the pipeline and scheduler.
