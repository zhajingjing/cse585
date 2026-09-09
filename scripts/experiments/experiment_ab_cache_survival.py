"""
Practical A/B survival experiment for Wan 2.1 latent reuse.

For a source prompt A and one or more target prompts B, this script generates:
  - A_full
  - B_full
  - B_from_step5
  - B_from_step10

It then measures how close each B variant is to:
  - A_full  ("A-ness dominates")
  - B_full  ("B-ness survives")

Distances:
  - pixel MSE over the full video tensor
  - pixel L2 over the full video tensor
  - optional LPIPS over sampled frames if `lpips` is installed

Example:
  python scripts/experiments/experiment_ab_cache_survival.py \
    --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
    --prompt_a "A cat walking on the street" \
    --prompt_b "A kitty walking along the sidewalk" \
    --output_dir ./ab_survival_out

  python scripts/experiments/experiment_ab_cache_survival.py \
    --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
    --prompt_list eval/teacache/vbench/VBench_200_semantic_similar.json \
    --a_index 0 \
    --b_indices 1,2,3,4 \
    --output_dir ./ab_survival_out
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAN_ROOT = os.path.join(PROJECT_ROOT, "Wan2.1")
SERVING_ROOT = os.path.join(PROJECT_ROOT, "scripts", "serving")
for import_path in (PROJECT_ROOT, WAN_ROOT, SERVING_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import wan  # noqa: E402
from eval.teacache.experiments.utils import read_prompt_list  # noqa: E402
from serving_system_video_teacache import (  # noqa: E402
    configure_wan_teacache,
    reset_wan_teacache_state,
    wan_generate_with_latent_cache,
)
from transformers import CLIPModel, CLIPProcessor  # noqa: E402
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS  # noqa: E402
from wan.utils.utils import cache_video  # noqa: E402

try:
    import lpips  # type: ignore
except Exception:
    lpips = None

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


@dataclass
class GenerationResult:
    video: torch.Tensor
    collected_latents: list
    elapsed_seconds: float


def summarize_prompt(prompt, max_len=80):
    prompt = " ".join(str(prompt).split())
    if len(prompt) <= max_len:
        return prompt
    return prompt[: max_len - 3] + "..."


def safe_name(text, max_len=80):
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned[:max_len] if cleaned else "prompt"


def parse_b_indices(text):
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def sample_frame_indices(num_frames, num_samples):
    if num_samples <= 0 or num_samples >= num_frames:
        return list(range(num_frames))
    return sorted(set(np.linspace(0, num_frames - 1, num_samples, dtype=int).tolist()))


def frame_tensor_to_pil(frame_tensor):
    frame = frame_tensor.detach().cpu().clamp(-1, 1).add(1).div(2)
    frame = (frame.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def compute_lpips_distance(video_a, video_b, frame_indices, device):
    if lpips is None:
        return None
    loss_fn = lpips.LPIPS(net="alex").to(device)
    scores = []
    with torch.no_grad():
        for frame_idx in frame_indices:
            img_a = video_a[:, frame_idx].unsqueeze(0).to(device=device, dtype=torch.float32)
            img_b = video_b[:, frame_idx].unsqueeze(0).to(device=device, dtype=torch.float32)
            scores.append(loss_fn(img_a, img_b).mean().item())
    return float(np.mean(scores)) if scores else None


def compute_clip_text_video_alignment(video, prompt, frame_indices, clip_model, clip_processor, device):
    images = [frame_tensor_to_pil(video[:, frame_idx]) for frame_idx in frame_indices]
    inputs = clip_processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        similarities = torch.matmul(image_embeds, text_embeds.T).squeeze(-1)
    return float(similarities.mean().item())


def save_video_tensor(video, save_path, fps):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cache_video(
        tensor=video[None],
        save_file=save_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )


def generate_video(
    pipeline,
    cfg,
    prompt,
    size,
    num_frames,
    sample_steps,
    sample_solver,
    sample_shift,
    guide_scale,
    seed,
    offload_model,
    collect_steps=None,
    cache_latent=None,
    cache_start_step=None,
):
    start_time = time.time()
    result = pipeline.generate(
        prompt,
        size=size,
        frame_num=num_frames,
        shift=sample_shift,
        sample_solver=sample_solver,
        sampling_steps=sample_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
        collect_latents_at_steps=collect_steps,
        cache_latent=cache_latent,
        cache_start_step=cache_start_step,
    )
    if isinstance(result, tuple):
        video, collected_latents = result
    else:
        video = result
        collected_latents = []
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Expected video tensor, got {type(video).__name__}")
    return GenerationResult(
        video=video.detach().cpu(),
        collected_latents=collected_latents,
        elapsed_seconds=time.time() - start_time,
    )


def build_pipeline(args):
    cfg = WAN_CONFIGS[args.task]
    pipeline = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    pipeline.__class__.generate = wan_generate_with_latent_cache

    if args.enable_teacache:
        configure_wan_teacache(
            pipeline.model,
            checkpoint_dir=args.ckpt_dir,
            sampling_steps=args.sample_steps,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
        )

    return cfg, pipeline


def maybe_reset_teacache(pipeline, args, sampling_steps):
    if args.enable_teacache:
        reset_wan_teacache_state(pipeline.model)
        configure_wan_teacache(
            pipeline.model,
            checkpoint_dir=args.ckpt_dir,
            sampling_steps=sampling_steps,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
        )


def run_pair_experiment(
    args,
    cfg,
    pipeline,
    clip_model,
    clip_processor,
    prompt_a,
    prompt_b,
    pair_dir,
    frame_indices,
):
    size = SIZE_CONFIGS[args.size]

    maybe_reset_teacache(pipeline, args, args.sample_steps)
    a_result = generate_video(
        pipeline,
        cfg,
        prompt_a,
        size,
        args.num_frames,
        args.sample_steps,
        args.sample_solver,
        args.sample_shift,
        args.guide_scale,
        args.seed,
        args.offload_model,
        collect_steps=(5, 10),
    )
    save_video_tensor(a_result.video, os.path.join(pair_dir, "A_full.mp4"), cfg.sample_fps)

    maybe_reset_teacache(pipeline, args, args.sample_steps)
    b_full_result = generate_video(
        pipeline,
        cfg,
        prompt_b,
        size,
        args.num_frames,
        args.sample_steps,
        args.sample_solver,
        args.sample_shift,
        args.guide_scale,
        args.seed,
        args.offload_model,
    )
    save_video_tensor(b_full_result.video, os.path.join(pair_dir, "B_full.mp4"), cfg.sample_fps)

    if len(a_result.collected_latents) != 2:
        raise ValueError(
            f"Expected latents for steps (5, 10), got {len(a_result.collected_latents)} items"
        )

    resumed_results = {}
    for cache_step, cached_latent in zip((5, 10), a_result.collected_latents):
        maybe_reset_teacache(pipeline, args, args.sample_steps - cache_step)
        resumed = generate_video(
            pipeline,
            cfg,
            prompt_b,
            size,
            args.num_frames,
            args.sample_steps,
            args.sample_solver,
            args.sample_shift,
            args.guide_scale,
            args.seed,
            args.offload_model,
            cache_latent=cached_latent,
            cache_start_step=cache_step,
        )
        name = f"B_from_step{cache_step}"
        resumed_results[name] = resumed
        save_video_tensor(resumed.video, os.path.join(pair_dir, f"{name}.mp4"), cfg.sample_fps)

    metric_rows = []
    for name, result in resumed_results.items():
        video = result.video
        lpips_to_a = compute_lpips_distance(video, a_result.video, frame_indices, pipeline.device)
        lpips_to_b = compute_lpips_distance(video, b_full_result.video, frame_indices, pipeline.device)
        clip_to_a = compute_clip_text_video_alignment(
            video, prompt_a, frame_indices, clip_model, clip_processor, pipeline.device
        )
        clip_to_b = compute_clip_text_video_alignment(
            video, prompt_b, frame_indices, clip_model, clip_processor, pipeline.device
        )
        metric_rows.append(
            {
                "pair_dir": pair_dir,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "variant": name,
                "time_A_full_seconds": a_result.elapsed_seconds,
                "time_B_full_seconds": b_full_result.elapsed_seconds,
                "time_variant_seconds": result.elapsed_seconds,
                "lpips_to_A_full": lpips_to_a,
                "lpips_to_B_full": lpips_to_b,
                "clip_align_to_prompt_A": clip_to_a,
                "clip_align_to_prompt_B": clip_to_b,
                "clip_semantic_margin_B_minus_A": clip_to_b - clip_to_a,
            }
        )

    return metric_rows


def resolve_prompts(args):
    if args.prompt_a and args.prompt_b:
        return args.prompt_a, [args.prompt_b]

    if not args.prompt_list:
        raise ValueError("Provide either --prompt_a/--prompt_b or --prompt_list")

    prompts = read_prompt_list(args.prompt_list)
    if not (0 <= args.a_index < len(prompts)):
        raise IndexError(f"a_index {args.a_index} out of range for prompt list of size {len(prompts)}")

    prompt_a = prompts[args.a_index]
    if args.prompt_b:
        return prompt_a, [args.prompt_b]

    b_indices = parse_b_indices(args.b_indices)
    if not b_indices:
        raise ValueError("When using --prompt_list, provide at least one index via --b_indices")

    prompt_bs = []
    for idx in b_indices:
        if not (0 <= idx < len(prompts)):
            raise IndexError(f"b_index {idx} out of range for prompt list of size {len(prompts)}")
        prompt_bs.append(prompts[idx])
    return prompt_a, prompt_bs


def main():
    parser = argparse.ArgumentParser(description="A/B survival experiment for Wan latent reuse")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to Wan checkpoint directory")
    parser.add_argument("--task", type=str, default="t2v-1.3B", choices=["t2v-1.3B", "t2v-14B"])
    parser.add_argument("--size", type=str, default="832*480", choices=sorted(SIZE_CONFIGS.keys()))
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--enable_teacache", action="store_true")
    parser.add_argument("--teacache_thresh", type=float, default=0.2)
    parser.add_argument("--use_ret_steps", action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--frame_samples", type=int, default=8, help="Frames sampled for LPIPS")
    parser.add_argument("--output_dir", type=str, default="./ab_cache_survival_outputs")

    parser.add_argument("--prompt_a", type=str, default=None)
    parser.add_argument("--prompt_b", type=str, default=None)
    parser.add_argument("--prompt_list", type=str, default=None)
    parser.add_argument("--a_index", type=int, default=0)
    parser.add_argument("--b_indices", type=str, default="1")
    args = parser.parse_args()

    if args.size not in SUPPORTED_SIZES[args.task]:
        raise ValueError(
            f"Unsupported size {args.size} for {args.task}; choose from {SUPPORTED_SIZES[args.task]}"
        )
    if args.sample_steps <= 10:
        raise ValueError("--sample_steps must be greater than 10 to resume from step 10")

    prompt_a, prompt_bs = resolve_prompts(args)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg, pipeline = build_pipeline(args)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(pipeline.device)
    clip_model.eval()
    frame_indices = sample_frame_indices(args.num_frames, args.frame_samples)

    metrics = []
    for idx, prompt_b in enumerate(prompt_bs):
        pair_label = f"pair_{idx:02d}_{safe_name(prompt_a, 30)}__{safe_name(prompt_b, 30)}"
        pair_dir = os.path.join(args.output_dir, pair_label)
        os.makedirs(pair_dir, exist_ok=True)

        print(f"[Experiment] A='{summarize_prompt(prompt_a)}'", flush=True)
        print(f"[Experiment] B='{summarize_prompt(prompt_b)}'", flush=True)

        rows = run_pair_experiment(
            args,
            cfg,
            pipeline,
            clip_model,
            clip_processor,
            prompt_a,
            prompt_b,
            pair_dir,
            frame_indices,
        )
        metrics.extend(rows)

        pair_csv = os.path.join(pair_dir, "metrics.csv")
        with open(pair_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_csv = os.path.join(args.output_dir, "summary_metrics.csv")
    if metrics:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)

    print(f"[Done] Wrote outputs and metrics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
