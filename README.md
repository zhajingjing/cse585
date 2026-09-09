# V-Nirvana: Serving Video Generation with Intra-User and Inter-User Cache Reuse

V-Nirvana accelerates Wan 2.1 text-to-video inference by combining two forms of cache reuse:

- **Intra-user reuse:** TeaCache skips redundant transformer computation within a request.
- **Inter-user reuse:** CLIP and FAISS identify similar prompts and reuse intermediate latents across requests.

The repository also includes VideoSys pipelines, evaluation utilities, profiling scripts, and experiment outputs.

## Requirements

- Linux with one or more CUDA-capable NVIDIA GPUs
- Python 3.10+
- A local Wan 2.1 T2V checkpoint

Dependencies are currently split between `requirements.txt` and `Wan2.1/requirements.txt` because the repository contains both VideoSys and Wan-based experiments. Use a dedicated environment and install the requirements needed by the workflow you run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r Wan2.1/requirements.txt
pip install faiss-cpu pandas
pip install -e . --no-deps
```

## Quick start

Run TeaCache with the cross-request latent cache:

```bash
python scripts/serving/serving_system_video_teacache.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_list eval/teacache/vbench/VBench_small_info.json \
  --video_directory ./video_outputs_teacache \
  --size '832*480' \
  --eval_mode
```

Useful comparisons:

```bash
# TeaCache only
python scripts/serving/serving_system_video_teacache.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_list eval/teacache/vbench/VBench_small_info.json \
  --video_directory ./video_outputs_teacache_only \
  --no_nirvana --eval_mode

# Plain Wan denoising
python scripts/serving/serving_system_video_teacache.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_list eval/teacache/vbench/VBench_small_info.json \
  --video_directory ./video_outputs_baseline \
  --no_nirvana --no_teacache --eval_mode
```

Run `python scripts/serving/serving_system_video_teacache.py --help` for all options.

## Main files

| Path | Purpose |
|---|---|
| `scripts/serving/` | Wan serving entry points and cache implementation |
| `scripts/experiments/` | Latent profiling and A/B reuse experiments |
| `scripts/analysis/` | A/B result aggregation and plotting |
| `scripts/*.sh` | Environment setup and experiment launchers |
| `eval/teacache/common_metrics/` | CLIP, LPIPS, PSNR, and SSIM video evaluation |
| `eval/teacache/profiler.py` | Serving metrics helper |
| `eval/teacache/` | Workloads and evaluation utilities |
| `videosys/` | VideoSys package |
| `Wan2.1/` | Embedded Wan 2.1 implementation |

More serving details are available in [`docs/README_serving_video_teacache.md`](docs/README_serving_video_teacache.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
