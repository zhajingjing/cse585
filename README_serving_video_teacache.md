# TeaCache Video Serving For Wan 2.1

This setup serves **Wan 2.1 text-to-video** with two layers of reuse:

- **Nirvana-style cross-request cache**: CLIP + FAISS chooses the nearest cached prompt and reuses an intermediate latent at steps `5/10/15`.
- **TeaCache within the active request**: once a request starts denoising, Wan TeaCache can skip repeated transformer work inside the remaining steps.

The important design choice is that **TeaCache state is reset at the start of every request**, including cache hits. Reusing a latent from another prompt is reasonable; reusing TeaCache residual history from another prompt is not.

## Run

Run from the project root:

```bash
python serving_system_video_teacache.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_list eval/teacache/vbench/VBench_full_info.json \
  --video_directory ./samples/wan_teacache_nirvana \
  --eval_mode
```

## Main Arguments

| Argument | Default | Notes |
|---|---:|---|
| `--ckpt_dir` | required | Path to Wan checkpoint directory |
| `--task` | `t2v-1.3B` | `t2v-1.3B` or `t2v-14B` |
| `--size` | `832*480` | For `t2v-1.3B`, use `480*832` or `832*480` |
| `--num_frames` | `81` | Wan expects `4n+1` |
| `--sample_steps` | `50` | Total denoising steps |
| `--sample_solver` | `unipc` | `unipc` or `dpm++` |
| `--sample_shift` | `5.0` | Wan scheduler shift |
| `--guide_scale` | `5.0` | CFG scale |
| `--teacache_thresh` | `0.2` | Higher means more TeaCache skipping |
| `--use_ret_steps` | off | Enables Wan retention-step TeaCache mode |
| `--offload_model` | off | Saves VRAM, slower |
| `--cache_size` | `1000` | Max cached requests; each stores 3 latent tiers |
| `--loop` | `5` | Videos per prompt |
| `--eval_mode` | off | Submit all requests immediately |
| `--no_nirvana` | off | Disable cross-request latent reuse |

## Examples

Quick run:

```bash
python serving_system_video_teacache.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --num_req 50 \
  --video_directory ./video_outputs_teacache
```

Custom size and frame count:

```bash
python serving_system_video_teacache.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --size 480*832 \
  --num_frames 33 \
  --video_directory ./custom_out
```

Compare with and without Nirvana cache:

```bash
python serving_system_video_teacache.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_list eval/teacache/vbench/VBench_full_info.json \
  --num_req 20 \
  --video_directory ./out_with_nirvana \
  --eval_mode

python serving_system_video_teacache.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_list eval/teacache/vbench/VBench_full_info.json \
  --num_req 20 \
  --video_directory ./out_no_nirvana \
  --eval_mode \
  --no_nirvana
```

## Behavior Summary

- Scheduler process:
  - Encodes prompts with CLIP.
  - Uses FAISS nearest-neighbor search over cached prompt embeddings.
  - Maps similarity to a reuse tier: `>0.95 -> 15`, `>0.85 -> 10`, `>0.65 -> 5`.
- Worker process:
  - **Miss**: runs full Wan generation, collects latents at `5/10/15`, and writes them into the shared cache.
  - **Hit**: loads the cached latent, resets TeaCache state, and continues denoising from step `k` to the final step.
- Cache:
  - Uses `KMinHeapCache` with LCBFU-style scoring.
  - Stores one latent per configured step tier.

## Notes

- If `--sample_steps <= 15`, the script rejects the run because the configured cache tiers would be invalid.
- Cache reuse only works when all requests use the same latent shape, so keep `--task`, `--size`, and `--num_frames` fixed within a run.
- On cache hits, generation is effectively resumed from the stored latent. That means changing the seed after the reuse point does not introduce new randomness the way a full fresh run would.
