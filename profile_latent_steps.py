"""
Profiling script: two experiments to find latent caching thresholds in Wan 2.1 T2V-1.3B.

Prompt categories are designed to test three orthogonal axes:
  - Semantic distance (near-duplicate → same category → unrelated)
  - Motion complexity (static → simple motion → causal/multi-event)
  - Temporal causality (prompts where frame N causes frame N+1)

Experiment 1 — Latent Similarity Curve:
  Run prompts from each category, collect latent z at every denoising step,
  compute pairwise cosine similarity. Shows when intra-group prompts diverge
  from each other vs inter-group pairs. Reveals which features are "locked in"
  early (global layout, motion type) vs late (colour, texture, fine detail).

Experiment 2 — Cross-Prompt Injection Test:
  Run prompt A fully, save intermediate latents at steps k=1..T.
  For prompt B, inject A's step-k latent via callback and continue with B's text.
  Plot L2(output, baseline_B) vs k — the inflection is your cache cutoff.
"""

import os
import torch
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm
# from transformers import XCLIPTokenizer, XCLIPTextModel
# from scipy.stats import spearmanr

import sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Wan2.1"))
from wan.text2video import WanT2V
from wan.configs import WAN_CONFIGS
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# ── Configuration ──────────────────────────────────────────────────────────────

# Path to the downloaded Wan2.1-T2V-1.3B checkpoint directory
CKPT_DIR   = "/root/autodl-tmp/Wan2.1-T2V-1.3B"
NUM_STEPS  = 50
HEIGHT     = 480
WIDTH      = 832
NUM_FRAMES = 33        # ~2 s at 16 fps
FPS        = 16
SEED       = 42
BATCH_SIZE = 2         # prompts per generate call (reduce if OOM)
OUTPUT_DIR = "./profile_outputs"

# Standard Wan negative prompt (from model card)
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, image, still, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)

# ── Prompt catalogue ───────────────────────────────────────────────────────────
#
# Each group tests a specific axis. Within a group, prompts are ordered from
# most-similar to most-different relative to the first entry.
#
# Group naming convention:
#   A  — attribute swap (colour/texture only changes)
#   B  — subject swap (same action, different animal/object)
#   C  — motion/dynamics swap (same subject, different motion type)
#   D  — scene/setting swap (same subject+motion, different environment)
#   E  — temporal-causal sequences (events that imply frame causality)
#   F  — unrelated control (maximally different scenes)

PROMPT_GROUPS = {
    # ── A: Colour/texture attribute swap ──────────────────────────────────────
    # Hypothesis: latents stay similar until very late steps when colour
    # details are resolved. Good cache candidates even at high step counts.
    "A_attribute": [
        ("A1", "a brown horse galloping across a green meadow"),
        ("A2", "a white horse galloping across a green meadow"),   # colour only
        ("A3", "a black horse galloping across a green meadow"),   # colour only
        ("A4", "a spotted horse galloping across a green meadow"), # texture change
    ],

    # ── B: Subject swap, same motion ─────────────────────────────────────────
    # Hypothesis: latents share global motion structure early but diverge
    # in mid steps as subject shape/silhouette is resolved.
    "B_subject": [
        ("B1", "a golden retriever running through a park"),
        ("B2", "a grey wolf running through a park"),   # different animal, same action
        ("B3", "a young child running through a park"), # human vs animal
        ("B4", "a red sports car driving through a park"), # object vs living
    ],

    # ── C: Motion/dynamics swap, same subject ────────────────────────────────
    # Hypothesis: motion type is encoded early (optical flow structure in latent).
    # Static vs dynamic prompts should diverge almost immediately at step 1-5.
    "C_motion": [
        ("C1", "a waterfall cascading down a rocky cliff"),        # continuous flow
        ("C2", "a waterfall frozen in ice on a rocky cliff"),      # static (no motion)
        ("C3", "a waterfall slowly trickling down a rocky cliff"), # slow motion
        ("C4", "a waterfall exploding outward from a rocky cliff"),# violent motion
    ],

    # ── D: Setting/environment swap ───────────────────────────────────────────
    # Hypothesis: background/lighting diverges mid steps; subject stays similar.
    "D_setting": [
        ("D1", "a surfer riding a wave at golden sunset"),
        ("D2", "a surfer riding a wave under grey storm clouds"), # lighting change
        ("D3", "a surfer riding a wave at night under moonlight"),# drastic lighting
        ("D4", "a surfer riding a wave in a tropical lagoon"),    # environment change
    ],

    # ── E: Temporal-causal sequences ─────────────────────────────────────────
    # These prompts explicitly describe events that unfold over time.
    # Key question: does the model encode the *causal structure* (which frame
    # causes which) early in denoising, or does it emerge late?
    # Strategy: compare pairs where the causal ORDER is swapped.
    "E_causal": [
        ("E1", "a match is struck and a candle flame grows brighter"),  # cause→effect
        ("E2", "a candle flame flickers and then goes out"),             # effect→resolution
        ("E3", "a glass of water tips over and spills across a table"),  # cause→effect
        ("E4", "a puddle of water on a table slowly evaporates"),        # reversed causality feel
        ("E5", "a tennis ball bounces off the ground and flies upward"), # physics: impact→rebound
        ("E6", "a tennis ball arcs downward and strikes the ground"),    # reversed arc
    ],

    # ── F: Unrelated control (maximally different) ────────────────────────────
    # Hypothesis: these should diverge at step 1. If they don't, the initial
    # noise dominates — useful calibration for how much seed matters.
    "F_control": [
        ("F1", "a brown horse galloping across a green meadow"),  # same as A1
        ("F2", "a rocket launching into a starry night sky"),
        ("F3", "microscopic cells dividing under a microscope"),
        ("F4", "a chef flipping pancakes in a busy kitchen"),
    ],
}

# Flat list for Experiment 1 (all prompts)
PROMPTS_EXP1 = [(gid, pid, text)
                for gid, entries in PROMPT_GROUPS.items()
                for pid, text in entries]

# ── Experiment 2: injection pairs ─────────────────────────────────────────────
# Each tuple: (label, source_prompt, target_prompt, expected_relationship)
INJECTION_PAIRS = [
    ("near_attr",  "a brown horse galloping across a green meadow",
                   "a white horse galloping across a green meadow",
                   "near-duplicate (colour swap) — expect late divergence"),
    ("mid_subj",   "a golden retriever running through a park",
                   "a grey wolf running through a park",
                   "same action, different subject — expect mid divergence"),
    ("causal_rev", "a tennis ball bounces off the ground and flies upward",
                   "a tennis ball arcs downward and strikes the ground",
                   "reversed causality — expect early divergence"),
    ("unrelated",  "a brown horse galloping across a green meadow",
                   "a rocket launching into a starry night sky",
                   "unrelated — expect immediate divergence"),
]

# Steps at which to save latents for injection (1-based, first 30 of 50)
INJECT_STEPS = [1, 5, 10, 15, 20, 25, 30, 40]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Build pipeline ─────────────────────────────────────────────────────────────
# Use the original Wan2.1 pipeline directly from the cloned repo.
# WanT2V loads T5, VAE, and WanModel (our modified version with CHAI support).

cfg      = WAN_CONFIGS["t2v-1.3B"]
pipeline = WanT2V(cfg, checkpoint_dir=CKPT_DIR, device_id=0, rank=0)
model    = pipeline.model          # WanModel — CHAI methods live here
vae      = pipeline.vae            # WanVAE
device   = torch.device("cuda:0")

# Latent shape constants derived from config
_VAE_STRIDE  = cfg.vae_stride                  # (4, 8, 8)
_PATCH_SIZE  = cfg.patch_size                  # (1, 2, 2)
_Z_DIM       = vae.model.z_dim                 # 16
_F_LAT       = (NUM_FRAMES - 1) // _VAE_STRIDE[0] + 1          # 9
_H_LAT       = HEIGHT  // _VAE_STRIDE[1]                        # 60
_W_LAT       = WIDTH   // _VAE_STRIDE[2]                        # 104
SEQ_LEN      = math.ceil((_H_LAT * _W_LAT) /
                          (_PATCH_SIZE[1] * _PATCH_SIZE[2]) *
                          _F_LAT)               # 14040


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_generator():
    return torch.Generator("cuda").manual_seed(SEED)


def _save_video(frames, path):
    """Save a list of PIL Images to an mp4."""
    imageio.mimwrite(path, [np.array(f, dtype=np.uint8) for f in frames], fps=FPS)


def _vae_decode_single(latent):
    """Decode one [C, F, H, W] latent (float32). Returns [C, F, H, W] in [-1, 1]."""
    return vae.decode([latent.to(device, dtype=torch.float32)])[0]  # [C, F, H, W]


def latent_to_thumb(z, size=(128, 72)):
    """Decode a single [C, F, H, W] latent to a PIL thumbnail of the first frame."""
    with torch.no_grad():
        video = _vae_decode_single(z)  # [C, F, H, W], [-1, 1]
    video = video.clamp(-1, 1).add(1).div(2)   # [0, 1]
    frame = video[:, 0].permute(1, 2, 0).float().cpu().numpy()
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(frame).resize(size, Image.LANCZOS)


def _build_phased_timesteps(full_timesteps, start_step,
                             micro_len=3, micro_subdivide=3):
    """
    Construct a non-uniform timestep array for post-injection denoising with the
    SAME total step count as the original remaining timesteps, but redistributed
    so early Δt is small and later Δt is large.

    Phase 2 — micro correction (first micro_len original intervals after injection):
        Each original interval [t_i → t_{i+1}] is subdivided into micro_subdivide
        equal sub-intervals. Uses n_micro * micro_subdivide steps total.

    Phase 3 — coarse steps (remainder of range):
        The remaining step budget (N - phase2_steps) is spread uniformly over
        [t_boundary → t_end], giving proportionally larger Δt. No steps are skipped.

    Total steps == len(remaining) - 1, identical to the original schedule.

    Args:
        full_timesteps : 1-D tensor from scheduler.set_timesteps, values high→low.
        start_step     : number of steps already done (injection index).
        micro_len      : number of original intervals in micro phase (default 3).
        micro_subdivide: sub-divisions per micro interval (default 3 → Δt/3).

    Returns 1-D float tensor — the complete timestep array to drive the loop.
    """
    remaining = full_timesteps[start_step:]   # values we still need to cover

    if start_step == 0 or len(remaining) <= micro_len + 1:
        return remaining

    N = len(remaining) - 1                          # total original intervals
    n_micro = min(micro_len, N)                     # how many intervals to subdivide
    phase2_steps = n_micro * micro_subdivide        # steps consumed by phase 2
    phase3_steps = N - phase2_steps                 # steps left for phase 3

    if phase3_steps <= 0:
        return remaining

    # ── Phase 2: dense sub-steps inside each micro interval ──────────────────
    phase2 = []
    for i in range(n_micro):
        t_hi = remaining[i].item()
        t_lo = remaining[i + 1].item()
        for sub in range(micro_subdivide):
            phase2.append(t_hi + (t_lo - t_hi) * (sub / micro_subdivide))
    t_boundary = remaining[n_micro].item()
    phase2.append(t_boundary)                       # exact boundary anchor

    # ── Phase 3: uniform steps consuming the leftover budget ─────────────────
    # Spread phase3_steps steps from t_boundary to t_end; Δt is larger to
    # compensate for the dense phase 2 — but NO steps are skipped overall.
    t_end = remaining[-1].item()
    phase3 = [t_boundary + (t_end - t_boundary) * (j / phase3_steps)
              for j in range(phase3_steps + 1)]

    # Combine — drop the shared boundary point that ends phase2 / starts phase3
    timesteps = torch.tensor(phase2 + phase3[1:],
                             dtype=remaining.dtype, device=remaining.device)

    # ── Print actual timestep values and Δt per phase ─────────────────────────
    p2 = phase2
    p3 = phase3
    dt_p2 = [f"{abs(p2[i+1]-p2[i]):.2f}" for i in range(len(p2)-1)]
    dt_p3 = [f"{abs(p3[i+1]-p3[i]):.2f}" for i in range(len(p3)-1)]
    print(f"  [phase2 timesteps] {[round(v,3) for v in p2]}")
    print(f"  [phase2 Δt]        {dt_p2}")
    print(f"  [phase3 timesteps] {[round(v,3) for v in p3]}")
    print(f"  [phase3 Δt]        {dt_p3}")
    print(f"  [summary] inject@{start_step} | "
          f"phase2: {len(p2)-1} steps (÷{micro_subdivide}) | "
          f"phase3: {len(p3)-1} steps (uniform stretch) | "
          f"total {len(timesteps)} == {len(remaining)-1} original remaining | "
          f"endpoint={timesteps[-1].item():.4f} (exact={remaining[-1].item():.4f})")
    return timesteps


def _make_scheduler(num_steps=NUM_STEPS):
    """Create a fresh FlowUniPCMultistepScheduler with standard Wan2.1 settings."""
    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    sched.set_timesteps(num_steps, device=device, shift=5.0)
    return sched


def _encode_text(prompts):
    """Encode a list of strings with T5. Returns list of [L, C] tensors on device."""
    pipeline.text_encoder.model.to(device)
    ctx = pipeline.text_encoder(prompts, device)
    pipeline.text_encoder.model.cpu()
    return ctx


def _generate(prompts, collect_steps=None, start_latent=None, start_step=None,
              micro_len=3, micro_subdivide=3, chai_inject=False, num_steps=NUM_STEPS):
    """
    Denoising loop using the original Wan2.1 WanModel directly.

      prompts      — str or list[str]
      collect_steps— 1-based step numbers at which to save the latent
      start_latent — Tensor[B, C, F, H, W] to resume from (Exp 2 injection)
      start_step   — int, how many steps already done (used with start_latent)
      chai_inject  — bool: use CHAI K/V injection on the conditional pass

    Returns:
      frames    — list[list[PIL.Image]], one inner list per prompt
      collected — list of Tensor[B, C, F, H, W] (cpu float32), one per step
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    B = len(prompts)
    collect_set = set(collect_steps) if collect_steps else set()
    step_offset = start_step if start_step is not None else 0
    collected   = []

    # 1. Text encoding
    context      = _encode_text(prompts)
    context_null = _encode_text([NEGATIVE_PROMPT] * B)

    # 2. Timestep schedule
    scheduler   = _make_scheduler(num_steps)
    full_ts     = scheduler.timesteps            # high → low, length NUM_STEPS
    if start_step is not None:
        timesteps = _build_phased_timesteps(
            full_ts, step_offset, micro_len=micro_len,
            micro_subdivide=micro_subdivide)
    else:
        timesteps = full_ts
        print(f"[_generate] full generation — {len(timesteps)} steps")

    # 3. Initialize latents: list of B tensors [C, F_lat, H_lat, W_lat]
    seed_g = _make_generator()
    if start_latent is not None:
        # Accept Tensor[B, C, F, H, W] from old Exp-2 API
        latents = [start_latent[j].to(device, dtype=torch.float32)
                   for j in range(start_latent.shape[0])]
    else:
        latents = [
            torch.randn(_Z_DIM, _F_LAT, _H_LAT, _W_LAT,
                        dtype=torch.float32, device=device, generator=seed_g)
            for _ in range(B)
        ]

    # 4. Denoising loop
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="denoising")):
            t_tensor = torch.stack([t] * B)   # [B]
            step_1based = i + 1               # 1-based denoising step index

            # CHAI injection rules (from paper):
            #   - Skip step 1: Q is pure Gaussian, not yet prompt-modulated
            #   - Only steps 2, 3, 4: no benefit beyond step 4
            #   - Only the FIRST transformer block: injecting across successive
            #     blocks within the same step adds noise
            do_chai = chai_inject and (2 <= step_1based <= 4)

            if do_chai:
                model.chai_inject_mode(True)   # block 0 only, per paper
            noise_preds_cond = model(
                latents, t=t_tensor, context=context, seq_len=SEQ_LEN)
            if do_chai:
                model.chai_inject_mode(False)

            # Unconditional pass — never inject CHAI (CFG stays target-only)
            noise_preds_null = model(
                latents, t=t_tensor, context=context_null, seq_len=SEQ_LEN)

            # CFG + latent update — process each prompt independently
            # (scheduler.step expects [1, C, F, H, W], matching WanT2V's loop)
            new_latents = []
            for j in range(B):
                np_j = noise_preds_null[j] + 5.0 * (
                    noise_preds_cond[j] - noise_preds_null[j])   # [C, F, H, W]

                if start_step is not None:
                    t_scale = float(cfg.num_train_timesteps)
                    t_next  = timesteps[i + 1] if i + 1 < len(timesteps) \
                              else torch.zeros_like(t)
                    dt      = (t_next - t).float() / t_scale
                    upd     = (latents[j].float() + np_j.float() * dt)
                else:
                    upd = scheduler.step(
                        np_j.float().unsqueeze(0), t, latents[j].float().unsqueeze(0),
                        return_dict=False, generator=seed_g)[0].squeeze(0).float()  # [C, F, H, W]

                new_latents.append(upd)

            latents = new_latents

            step_num = step_offset + i + 1
            if step_num in collect_set:
                collected.append(torch.stack(latents).float().cpu().clone())  # [B, C, F, H, W]

    # 5. Decode latents → PIL frames
    all_frames = []
    with torch.no_grad():
        for j in range(B):
            video = _vae_decode_single(latents[j].float())   # [C, F, H, W], [-1,1]
            video = video.clamp(-1, 1).add(1).div(2)         # [0, 1]
            video = (video.permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
            all_frames.append([Image.fromarray(frame) for frame in video])

    return all_frames, collected


def _generate_chai(target_prompts, reference_prompt: str, num_steps=NUM_STEPS,
                   **generate_kwargs):
    """
    CHAI (CacHe Attention Inference) generation.

    Steps:
      1. Run reference_prompt for 50 steps → collect final output latent z_ref.
      2. Patch-embed z_ref → store hidden states on block 0's self-attention.
      3. Run target_prompts for num_steps; at steps 2,3,4 block 0 uses
         z_ref hidden states as K/V source (Q from target's current hidden state).
    """
    print(f"\n[CHAI] Step 1 — generate reference (50 steps): '{reference_prompt}'")
    _, ref_lat_list = _generate(reference_prompt, collect_steps=[NUM_STEPS],
                                num_steps=NUM_STEPS)
    z_ref = ref_lat_list[0][0]   # [C, F, H, W], cpu float32

    print("[CHAI] Step 2 — capture final transformer hidden state from reference")
    t_final      = _make_scheduler().timesteps[-1]
    context_ref  = _encode_text([reference_prompt])
    model.chai_capture_ref_hidden(
        x_ref=[z_ref.to(device, dtype=torch.bfloat16)],
        t_ref=torch.stack([t_final]).to(device),
        context_ref=context_ref,
        seq_len=SEQ_LEN,
    )

    print(f"[CHAI] Step 3 — generate target ({num_steps} steps) with injection at steps 2,3,4")
    try:
        return _generate(target_prompts, chai_inject=True, num_steps=num_steps,
                         **generate_kwargs)
    finally:
        model.chai_clear()   # remove stored hidden states


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4 — CHAI Cross-Prompt Attention Steering
# ══════════════════════════════════════════════════════════════════════════════

# Pairs: (label, reference_prompt, target_prompt)
# Reference provides the structural/visual anchor (K, V from its final latent).
# Target text drives semantics. Outputs are compared to the target baseline.
CHAI_PAIRS = [
    ("colour_swap",
     "a white horse galloping across a green meadow",
     "a brown horse galloping across a green meadow"),
    ("subject_swap",
     "a golden retriever running through a park",
     "a grey wolf running through a park"),
    ("motion_transfer",
     "a waterfall cascading down a rocky cliff",
     "a waterfall frozen in ice on a rocky cliff"),
    ("unrelated",
     "a brown horse galloping across a green meadow",
     "a rocket launching into a starry night sky"),
]


def run_exp4_chai():
    """
    Experiment 4: CHAI-style self-attention injection.

    For each (reference, target) pair:
      - Baseline: generate target from scratch.
      - CHAI:     generate target with K, V from reference final latent.
      - Compute per-frame L2 and cosine similarity between CHAI and baseline
        first frames to quantify how much the reference structural prior bleeds
        into the target output.

    Saves side-by-side comparison videos and a summary bar chart.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: CHAI Cross-Prompt Attention Steering")
    print("=" * 60)

    exp4_dir = os.path.join(OUTPUT_DIR, "exp4_chai")
    os.makedirs(exp4_dir, exist_ok=True)

    results = {}  # label → {"l2": float, "cos": float}

    for label, ref_prompt, tgt_prompt in CHAI_PAIRS:
        pair_dir = os.path.join(exp4_dir, label)
        os.makedirs(pair_dir, exist_ok=True)

        print(f"\n  [{label}]")
        print(f"    Reference: '{ref_prompt}'")
        print(f"    Target:    '{tgt_prompt}'")

        # Baseline — 8-step target generation without CHAI
        print("    → Baseline generation (8 steps) …")
        baseline_frames, baseline_lats = _generate(tgt_prompt, num_steps=8,
                                                    collect_steps=[8])
        _save_video(baseline_frames[0], os.path.join(pair_dir, "baseline.mp4"))

        # CHAI — 8-step target steered by reference K, V (from 50-step ref latent)
        print("    → CHAI generation (8 steps) …")
        chai_frames, chai_lats = _generate_chai(tgt_prompt, reference_prompt=ref_prompt,
                                                num_steps=8, collect_steps=[8])
        _save_video(chai_frames[0], os.path.join(pair_dir, "chai.mp4"))

        # ── Metrics ──────────────────────────────────────────────────────────
        # Latent-space cosine similarity: compare final latents (before decode)
        # More meaningful than pixel cosine — captures structure, not just colour.
        b_lat = baseline_lats[0][0].flatten().float()   # [C*F*H*W]
        c_lat = chai_lats[0][0].flatten().float()
        lat_cos = torch.nn.functional.cosine_similarity(
            b_lat.unsqueeze(0), c_lat.unsqueeze(0)).item()

        # Pixel L2 on first frame (how visually different the outputs look)
        b_px = np.array(baseline_frames[0][0], dtype=np.float32).flatten()
        c_px = np.array(chai_frames[0][0],     dtype=np.float32).flatten()
        px_l2 = float(np.linalg.norm(c_px - b_px))

        results[label] = {"lat_cos": lat_cos, "px_l2": px_l2}
        print(f"    latent cosine(CHAI vs baseline)={lat_cos:.4f}  "
              f"pixel L2={px_l2:.1f}")

    # ── Summary bar chart ─────────────────────────────────────────────────────
    labels   = list(results.keys())
    cos_vals = [results[lb]["lat_cos"] for lb in labels]
    l2_vals  = [results[lb]["px_l2"]  for lb in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))

    ax1.bar(x, cos_vals, color="steelblue")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Latent cosine similarity  (CHAI vs 8-step baseline)")
    ax1.set_title("Latent-space alignment\n1 = identical to baseline, lower = more reference bleed-in")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, l2_vals, color="darkorange")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Pixel L2  (CHAI vs 8-step baseline, first frame)")
    ax2.set_title("Visual deviation from baseline\nHigher = more reference influence on output")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment 4: CHAI cross-prompt self-attention injection\n"
                 "Ref K/V from 50-step final latent → injected into 8-step target self-attention",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(exp4_dir, "chai_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved CHAI summary chart → {path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Latent Similarity Curve
# ══════════════════════════════════════════════════════════════════════════════

def run_exp1():
    """
    For every prompt in PROMPT_GROUPS, collect latent z at every denoising step.
    Then plot two figures:
      (a) Per-group intra-similarity curves — shows when attributes split within a group.
      (b) Cross-group comparison — shows how different categories diverge.
    Also saves raw similarity matrix and L2 table to disk.
    """
    print("\n" + "="*60)
    print("Experiment 1: Latent Similarity Curve (grouped)")
    print("="*60)

    # all_latents: { pid: [z_step1, z_step2, ...] }  each z is [C, T, H, W]
    all_latents = {}
    pid_to_group = {}
    pid_to_text  = {}

    # Flatten all prompts and batch for speed.
    flat_entries = [(gid, pid, text)
                    for gid, entries in PROMPT_GROUPS.items()
                    for pid, text in entries]

    for batch_start in range(0, len(flat_entries), BATCH_SIZE):
        batch   = flat_entries[batch_start: batch_start + BATCH_SIZE]
        b_pids  = [e[1] for e in batch]
        b_texts = [e[2] for e in batch]
        print(f"\n  Generating batch {batch_start // BATCH_SIZE + 1}: {', '.join(b_pids)}")

        _, collected = _generate(b_texts, collect_steps=range(1, NUM_STEPS + 1))

        # collected: list[Tensor[B, C, T, H, W]], one tensor per step — split per prompt
        for i, (gid, pid, text) in enumerate(batch):
            all_latents[pid] = [z[i].float().cpu() for z in collected]
            pid_to_group[pid] = gid
            pid_to_text[pid]  = text
        print(f"    → {len(collected)} steps, shape per prompt {collected[0][0].shape}")

    pids      = list(all_latents.keys())
    n         = len(pids)
    steps     = len(all_latents[pids[0]])
    step_axis = list(range(1, steps + 1))

    # ── Compute pairwise cosine similarity ──────────────────────────────────
    sim = np.zeros((n, n, steps))
    for i in range(n):
        for j in range(n):
            for s in range(steps):
                zi = all_latents[pids[i]][s].flatten()
                zj = all_latents[pids[j]][s].flatten()
                sim[i, j, s] = torch.nn.functional.cosine_similarity(
                    zi.unsqueeze(0), zj.unsqueeze(0)).item()

    np.save(os.path.join(OUTPUT_DIR, "exp1_sim_matrix.npy"), sim)
    with open(os.path.join(OUTPUT_DIR, "exp1_pid_list.txt"), "w") as f:
        for pid in pids:
            f.write(f"{pid}\t{pid_to_group[pid]}\t{pid_to_text[pid]}\n")

    # ── Figure (a): intra-group similarity per group ─────────────────────────
    group_ids = list(PROMPT_GROUPS.keys())
    ncols = 3
    nrows = (len(group_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax_idx, gid in enumerate(group_ids):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        g_pids = [pid for pid, grp in pid_to_group.items() if grp == gid]
        for i in range(len(g_pids)):
            for j in range(i + 1, len(g_pids)):
                pi, pj = g_pids[i], g_pids[j]
                ii, jj = pids.index(pi), pids.index(pj)
                ax.plot(step_axis, sim[ii, jj], linewidth=1.5, label=f"{pi} vs {pj}")
        ax.set_title(gid, fontsize=9)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Cosine similarity")
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.25)
    for ax_idx in range(len(group_ids), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)
    fig.suptitle("Intra-group latent cosine similarity vs denoising step\n"
                 "(dashed = 0.9 reuse threshold)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "exp1_intragroup_similarity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved intra-group similarity → {path}")

    # ── Figure (b): cross-group representative pairs ──────────────────────────
    rep_pids = [entries[0][0] for entries in PROMPT_GROUPS.values()]
    rep_n    = len(rep_pids)
    fig, ax  = plt.subplots(figsize=(11, 6))
    colors   = plt.cm.tab10(np.linspace(0, 1, rep_n * (rep_n - 1) // 2))
    c_idx    = 0
    for i in range(rep_n):
        for j in range(i + 1, rep_n):
            pi, pj = rep_pids[i], rep_pids[j]
            ii, jj = pids.index(pi), pids.index(pj)
            gi, gj = pid_to_group[pi], pid_to_group[pj]
            ax.plot(step_axis, sim[ii, jj], linewidth=1.5,
                    label=f"{gi}[{pi}] vs {gj}[{pj}]", color=colors[c_idx])
            c_idx += 1
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Denoising step (1 = first, {NUM_STEPS} = last)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cross-category latent similarity (representative prompts)\n"
                 "Lower = categories diverged at that step")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "exp1_crossgroup_similarity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cross-group similarity → {path}")

    # ── Figure (c): temporal-causal group — highlight causal vs reversed pairs ─
    e_pids = [pid for pid, grp in pid_to_group.items() if grp == "E_causal"]
    if len(e_pids) >= 6:
        causal_pairs = [(e_pids[0], e_pids[1]), (e_pids[2], e_pids[3]), (e_pids[4], e_pids[5])]
        fig, ax = plt.subplots(figsize=(9, 5))
        for pi, pj in causal_pairs:
            ii, jj = pids.index(pi), pids.index(pj)
            ax.plot(step_axis, sim[ii, jj], linewidth=2,
                    label=f"{pi}↔{pj}  ({pid_to_text[pi][:30]}…)")
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Temporal-causal pairs: causal vs time-reversed prompt similarity\n"
                     "If similarity stays high → model doesn't encode causal order early")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "exp1_causal_pairs.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved causal-pair similarity → {path}")

    # ── Print L2 table ────────────────────────────────────────────────────────
    print("\n  L2 distances (step 1 / mid / final) — same-group pairs:")
    for gid, entries in PROMPT_GROUPS.items():
        g_pids = [pid for pid, _ in entries]
        for i in range(len(g_pids)):
            for j in range(i + 1, len(g_pids)):
                pi, pj = g_pids[i], g_pids[j]
                l2s = [(all_latents[pi][s] - all_latents[pj][s]).norm().item()
                       for s in range(steps)]
                print(f"  [{gid}] {pi} vs {pj}: "
                      f"step1={l2s[0]:.2f}  mid={l2s[steps//2]:.2f}  final={l2s[-1]:.2f}")

    return all_latents, pid_to_group, pid_to_text


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Cross-Prompt Injection Test (all INJECTION_PAIRS)
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_injection_pair(label, prompt_source, prompt_target, description):
    """Run the injection test for one (source, target) pair. Returns L2 dict."""
    pair_dir = os.path.join(OUTPUT_DIR, f"exp2_{label}")
    os.makedirs(pair_dir, exist_ok=True)

    print(f"\n  Pair [{label}]: {description}")
    print(f"    Source: '{prompt_source}'")
    print(f"    Target: '{prompt_target}'")

    # 1. Collect source latents + source reference video in one pass.
    #    collected[i] has shape [1, C, T, H, W] — one latent per INJECT_STEPS entry.
    source_frames, source_latents = _generate(prompt_source, collect_steps=INJECT_STEPS)
    _save_video(source_frames[0], os.path.join(pair_dir, "source_reference.mp4"))

    # 2. Baseline: target from scratch (no injection)
    baseline_frames, _ = _generate(prompt_target)
    _save_video(baseline_frames[0], os.path.join(pair_dir, "baseline_target.mp4"))
    baseline_first = np.array(baseline_frames[0][0], dtype=np.float32)

    # 3. Injection sweep: for each step k, replace target's latent at step k
    #    with source's step-k latent, then continue with target's text.
    latent_distances = {}
    for idx, inject_at_step in enumerate(INJECT_STEPS):
        # Reuse the first inject_at_step steps from source by starting from
        # the source's latent at that step and only running the remaining steps.
        inj_frames, _ = _generate(
            prompt_target,
            start_latent=source_latents[idx],   # shape [1, C, T, H, W]
            start_step=inject_at_step,           # skip steps 1..inject_at_step
        )
        _save_video(inj_frames[0],
                    os.path.join(pair_dir, f"inject_step{inject_at_step:03d}.mp4"))

        inj_first = np.array(inj_frames[0][0], dtype=np.float32)
        l2 = np.linalg.norm(inj_first - baseline_first)
        latent_distances[inject_at_step] = l2
        print(f"      step={inject_at_step:3d}  L2={l2:.1f}")

    return latent_distances


def run_exp2():
    print("\n" + "="*60)
    print("Experiment 2: Cross-Prompt Injection Test (all pairs)")
    print("="*60)

    all_results = {}
    for label, src, tgt, desc in INJECTION_PAIRS:
        all_results[label] = _run_one_injection_pair(label, src, tgt, desc)

    # ── Combined plot: all pairs on one figure ───────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(INJECTION_PAIRS)))
    for (label, _, _, desc), color in zip(INJECTION_PAIRS, colors):
        ld = all_results[label]
        steps_sorted = sorted(ld.keys())
        l2_vals = [ld[s] for s in steps_sorted]
        l2_arr  = np.array(l2_vals, dtype=float)
        l2_norm = (l2_arr - l2_arr.min()) / (l2_arr.max() - l2_arr.min() + 1e-6)
        ax.plot(steps_sorted, l2_norm, "o-", linewidth=2, markersize=5,
                color=color, label=f"[{label}] {desc[:55]}")

    ax.set_xlabel("Step at which source latent was injected")
    ax.set_ylabel("Normalised L2 distance to target baseline\n"
                  "(0 = identical to target, 1 = maximally distorted)")
    ax.set_title("Cross-Prompt Injection: when does source latent stop dominating output?\n"
                 "(curves should fall from 1→0; the knee = cache threshold)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "exp2_all_pairs_injection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved combined injection curve → {path}")

    # ── Per-pair summary ─────────────────────────────────────────────────────
    print("\n  Summary — estimated cache threshold per pair:")
    for label, src, tgt, desc in INJECTION_PAIRS:
        ld = all_results[label]
        steps_sorted = sorted(ld.keys())
        l2_vals = [ld[s] for s in steps_sorted]
        min_l2 = min(l2_vals)
        threshold_step = next(
            (s for s, l2 in zip(steps_sorted, l2_vals) if l2 < min_l2 * 1.15),
            steps_sorted[-1])
        print(f"    [{label}] {desc[:50]}")
        print(f"      step1 L2={l2_vals[0]:.1f}  final L2={l2_vals[-1]:.1f}  "
              f"→ safe from step ~{threshold_step}")

    return all_results


# # ══════════════════════════════════════════════════════════════════════════════
# # Experiment 3 — CLIP Text Similarity vs Denoising Latent Similarity
# # ══════════════════════════════════════════════════════════════════════════════

# VIDEOCLIP_MODEL_ID = "microsoft/xclip-base-patch32"

# def _encode_prompts_videoclip(prompts):
#     """Encode prompts with X-CLIP (video-language CLIP) text encoder. Returns L2-normalised [N, D]."""
#     tokenizer = XCLIPTokenizer.from_pretrained(VIDEOCLIP_MODEL_ID)
#     model = XCLIPTextModel.from_pretrained(VIDEOCLIP_MODEL_ID).to("cuda").eval()
#     with torch.no_grad():
#         tokens = tokenizer(prompts, padding=True, truncation=True,
#                            max_length=77, return_tensors="pt").to("cuda")
#         embeds = model(**tokens).pooler_output          # [N, D]
#         embeds = embeds / embeds.norm(dim=-1, keepdim=True)
#     model.cpu()
#     return embeds.float().cpu()


# def run_exp3(exp1_sim_path=None, exp1_pid_path=None):
#     """
#     Experiment 3: CLIP text similarity vs denoising latent similarity.

#     Two outputs:
#       (a) CLIP text similarity heatmap over all prompts, grouped by category.
#       (b) If exp1 data is available, Spearman correlation between the text
#           similarity matrix and the latent similarity matrix at each denoising
#           step — reveals at which step the video latent best reflects text semantics.

#     Args:
#       exp1_sim_path: path to exp1_sim_matrix.npy (optional, from run_exp1)
#       exp1_pid_path: path to exp1_pid_list.txt   (optional, from run_exp1)
#     """
#     print("\n" + "="*60)
#     print("Experiment 3: CLIP Text Similarity")
#     print("="*60)

#     # Collect all (pid, text) in the same order used by exp1
#     all_entries = [(pid, text)
#                    for entries in PROMPT_GROUPS.values()
#                    for pid, text in entries]
#     pids   = [e[0] for e in all_entries]
#     texts  = [e[1] for e in all_entries]
#     n      = len(pids)
#     pid_to_group = {pid: gid
#                     for gid, entries in PROMPT_GROUPS.items()
#                     for pid, _ in entries}

#     # ── 1. VideoCLIP (X-CLIP) text embeddings ───────────────────────────────
#     print(f"  Encoding {n} prompts with X-CLIP ({VIDEOCLIP_MODEL_ID})…")
#     embeds = _encode_prompts_videoclip(texts)     # [N, D], already L2-normalised

#     # Pairwise cosine similarity (dot product of unit vectors)
#     text_sim = (embeds @ embeds.T).numpy()        # [N, N]

#     # ── 2. Text similarity heatmap ───────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.4)))
#     im = ax.imshow(text_sim, vmin=-0.1, vmax=1.0, cmap="RdYlGn", aspect="auto")
#     plt.colorbar(im, ax=ax, label="Cosine similarity")
#     ax.set_xticks(range(n)); ax.set_xticklabels(pids, rotation=90, fontsize=7)
#     ax.set_yticks(range(n)); ax.set_yticklabels(pids, fontsize=7)

#     # Draw group boundary lines
#     group_sizes = [len(entries) for entries in PROMPT_GROUPS.values()]
#     boundaries = np.cumsum(group_sizes[:-1]) - 0.5
#     for b in boundaries:
#         ax.axhline(b, color="black", linewidth=1.2)
#         ax.axvline(b, color="black", linewidth=1.2)

#     # Annotate group labels on diagonal blocks
#     cursor = 0
#     for gid, entries in PROMPT_GROUPS.items():
#         mid = cursor + len(entries) / 2 - 0.5
#         ax.text(mid, -1.2, gid, ha="center", va="bottom", fontsize=6,
#                 color="navy", fontweight="bold")
#         cursor += len(entries)

#     ax.set_title("X-CLIP (VideoCLIP) text embedding cosine similarity\n(grouped by prompt category)")
#     plt.tight_layout()
#     path = os.path.join(OUTPUT_DIR, "exp3_clip_text_similarity.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved text similarity heatmap → {path}")

#     # Save raw matrix
#     np.save(os.path.join(OUTPUT_DIR, "exp3_text_sim_matrix.npy"), text_sim)

#     # ── 3. Correlation with exp1 latent similarities (if available) ──────────
#     sim_path = exp1_sim_path or os.path.join(OUTPUT_DIR, "exp1_sim_matrix.npy")
#     pid_path = exp1_pid_path or os.path.join(OUTPUT_DIR, "exp1_pid_list.txt")

#     if not (os.path.exists(sim_path) and os.path.exists(pid_path)):
#         print("  [skip] exp1 data not found — run run_exp1() first for correlation plot.")
#         return

#     lat_sim = np.load(sim_path)   # [N, N, steps]
#     with open(pid_path) as f:
#         exp1_pids = [line.split("\t")[0] for line in f if line.strip()]

#     # Align ordering: exp1 may have a different pid order
#     try:
#         idx = [exp1_pids.index(p) for p in pids]
#     except ValueError as e:
#         print(f"  [skip] pid mismatch between exp1 and current PROMPT_GROUPS: {e}")
#         return
#     lat_sim = lat_sim[np.ix_(idx, idx)]   # reorder to match current pids

#     steps = lat_sim.shape[2]
#     # Upper-triangle mask (exclude diagonal — always 1.0 vs 1.0)
#     mask = np.triu(np.ones((n, n), dtype=bool), k=1)
#     text_vec = text_sim[mask]             # [N*(N-1)/2]

#     correlations = []
#     for s in range(steps):
#         lat_vec = lat_sim[:, :, s][mask]
#         rho, _ = spearmanr(text_vec, lat_vec)
#         correlations.append(rho)

#     # ── Plot: Spearman ρ vs denoising step ───────────────────────────────────
#     step_axis = list(range(1, steps + 1))
#     peak_step = int(np.argmax(correlations)) + 1
#     peak_rho  = correlations[peak_step - 1]

#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(step_axis, correlations, linewidth=2, color="steelblue")
#     ax.axvline(peak_step, color="tomato", linestyle="--", linewidth=1.5,
#                label=f"peak ρ={peak_rho:.3f} at step {peak_step}")
#     ax.set_xlabel(f"Denoising step (1 = highest noise, {steps} = final)")
#     ax.set_ylabel("Spearman ρ  (text sim vs latent sim)")
#     ax.set_title("How well does CLIP text similarity predict video latent similarity?\n"
#                  "Peak = step where latent space most faithfully mirrors text semantics")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     path = os.path.join(OUTPUT_DIR, "exp3_text_latent_correlation.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved text–latent correlation curve → {path}")
#     print(f"  Peak alignment: step {peak_step}  (ρ = {peak_rho:.3f})")

#     # ── Per-group breakdown ───────────────────────────────────────────────────
#     # Within each group: does within-group text sim rank match within-group latent rank?
#     print("\n  Per-group CLIP vs latent correlation (at peak step):")
#     for gid, entries in PROMPT_GROUPS.items():
#         g_pids = [p for p, _ in entries]
#         if len(g_pids) < 3:
#             continue
#         g_idx  = [pids.index(p) for p in g_pids]
#         g_mask = np.triu(np.ones((len(g_idx), len(g_idx)), bool), k=1)
#         g_text = text_sim[np.ix_(g_idx, g_idx)][g_mask]
#         g_lat  = lat_sim[np.ix_(g_idx, g_idx), peak_step - 1].squeeze()[g_mask]
#         if g_text.std() < 1e-6:
#             print(f"    [{gid}] skipped (zero variance in text sim)")
#             continue
#         rho, _ = spearmanr(g_text, g_lat)
#         print(f"    [{gid}]  ρ = {rho:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["1", "2", "3", "4", "both"], default="both",
                        help="1=similarity curve, 2=injection test, 3=clip text sim, 4=chai attention, both=run all")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Subset of group IDs to run for exp1, e.g. --groups A_attribute E_causal")
    args = parser.parse_args()

    if args.groups:
        for gid in args.groups:
            assert gid in PROMPT_GROUPS, f"Unknown group '{gid}'. Valid: {list(PROMPT_GROUPS)}"
        original = PROMPT_GROUPS.copy()
        PROMPT_GROUPS.clear()
        for gid in args.groups:
            PROMPT_GROUPS[gid] = original[gid]

    if args.exp in ("1", "both"):
        run_exp1()
    if args.exp in ("2", "both"):
        run_exp2()
    # if args.exp in ("3", "both"):
    #     run_exp3()
    if args.exp in ("4", "both"):
        run_exp4_chai()

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
