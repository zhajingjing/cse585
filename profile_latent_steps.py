"""
Profiling script: two experiments to find latent caching thresholds in Open Sora.

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
  For prompt B, inject A's step-k latent and continue with B's text.
  Plot L2(output, baseline_B) vs k — the inflection is your cache cutoff.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from videosys import VideoSysEngine
from videosys.pipelines.open_sora.pipeline_open_sora import OpenSoraConfig

# ── Configuration ──────────────────────────────────────────────────────────────

NUM_STEPS = 30          # must match OpenSoraConfig.num_sampling_steps
RESOLUTION = "480p"
ASPECT_RATIO = "9:16"
NUM_FRAMES = "2s"
SEED = 42
OUTPUT_DIR = "./profile_outputs"

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

# Steps at which to save latents for injection
# Full: range(1, NUM_STEPS+1). Coarse for quick runs:
INJECT_STEPS = [1, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 27, 29, 30]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Build pipeline (no PAB, no TeaCache — clean baseline) ──────────────────────

config = OpenSoraConfig(
    num_sampling_steps=NUM_STEPS,
    cfg_scale=7.0,
    num_gpus=1,
    enable_pab=False,
)
engine = VideoSysEngine(config)
pipeline = engine.driver_worker.pipeline  # direct pipeline access


# ── Helper: decode one latent to a PIL thumbnail ──────────────────────────────

def latent_to_thumb(z, pipeline, num_frames_int=34, size=(128, 72)):
    """Decode a single latent tensor to a small PIL image (first frame)."""
    with torch.no_grad():
        video = pipeline.vae(z.to(pipeline._dtype), decode_only=True, num_frames=num_frames_int)
    video = video.clamp(-1, 1)
    video = (video + 1) / 2  # [0, 1]
    # video shape: [B, C, T, H, W] → take first batch, first frame
    frame = video[0, :, 0].permute(1, 2, 0).float().cpu().numpy()
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(frame).resize(size, Image.LANCZOS)
    return img


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

    # all_latents: { pid: [z_step0, z_step1, ...] }
    all_latents = {}
    pid_to_group = {}
    pid_to_text  = {}

    for group_id, entries in PROMPT_GROUPS.items():
        for pid, text in entries:
            print(f"\n  [{group_id}] Generating {pid}: '{text}'")
            _, collected = pipeline.generate(
                prompt=text,
                resolution=RESOLUTION,
                aspect_ratio=ASPECT_RATIO,
                num_frames=NUM_FRAMES,
                seed=SEED,
                verbose=False,
                collect_latents_at_steps=tuple(range(1, NUM_STEPS + 1)),
            )
            all_latents[pid] = [z.float().cpu() for z in collected]
            pid_to_group[pid] = group_id
            pid_to_text[pid]  = text
            print(f"    → {len(collected)} latents, shape {collected[0].shape}")

    pids  = list(all_latents.keys())
    n     = len(pids)
    steps = len(all_latents[pids[0]])
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
                label = f"{pi} vs {pj}"
                ax.plot(step_axis, sim[ii, jj], linewidth=1.5, label=label)
        ax.set_title(gid, fontsize=9)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Cosine similarity")
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.25)
    # hide unused axes
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
    # Pick the first prompt from each group as representative; plot all cross-pairs.
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
            label  = f"{gi}[{pi}] vs {gj}[{pj}]"
            ax.plot(step_axis, sim[ii, jj], linewidth=1.5, label=label, color=colors[c_idx])
            c_idx += 1
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Denoising step (1 = first, 30 = last)")
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
    # Group E: pairs (E1,E2), (E3,E4), (E5,E6) are causal vs reversed.
    e_pids = [pid for pid, grp in pid_to_group.items() if grp == "E_causal"]
    if len(e_pids) >= 6:
        causal_pairs = [(e_pids[0], e_pids[1]), (e_pids[2], e_pids[3]), (e_pids[4], e_pids[5])]
        fig, ax = plt.subplots(figsize=(9, 5))
        for pi, pj in causal_pairs:
            ii, jj = pids.index(pi), pids.index(pj)
            label = f"{pi}↔{pj}  ({pid_to_text[pi][:30]}…)"
            ax.plot(step_axis, sim[ii, jj], linewidth=2, label=label)
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

    # 1. Collect source latents
    _, source_latents = pipeline.generate(
        prompt=prompt_source,
        resolution=RESOLUTION, aspect_ratio=ASPECT_RATIO,
        num_frames=NUM_FRAMES, seed=SEED, verbose=False,
        collect_latents_at_steps=tuple(INJECT_STEPS),
    )

    # 2. Baseline: target from scratch
    baseline_result = pipeline.generate(
        prompt=prompt_target,
        resolution=RESOLUTION, aspect_ratio=ASPECT_RATIO,
        num_frames=NUM_FRAMES, seed=SEED, verbose=False,
    )
    baseline_video = baseline_result.video[0]
    Image.fromarray(baseline_video[0].numpy()).save(
        os.path.join(pair_dir, "baseline_target.png"))

    # 3. Source reference
    source_result = pipeline.generate(
        prompt=prompt_source,
        resolution=RESOLUTION, aspect_ratio=ASPECT_RATIO,
        num_frames=NUM_FRAMES, seed=SEED, verbose=False,
    )
    Image.fromarray(source_result.video[0][0].numpy()).save(
        os.path.join(pair_dir, "source_reference.png"))

    # 4. Injection sweep
    latent_distances = {}
    for idx, inject_at_step in enumerate(INJECT_STEPS):
        result_inj = pipeline.generate(
            prompt=prompt_target,
            resolution=RESOLUTION, aspect_ratio=ASPECT_RATIO,
            num_frames=NUM_FRAMES, seed=SEED, verbose=False,
            cache_latent=source_latents[idx].clone(),
            cache_start_step=inject_at_step,
        )
        first_frame = result_inj.video[0][0].numpy()
        Image.fromarray(first_frame.astype(np.uint8)).save(
            os.path.join(pair_dir, f"inject_step{inject_at_step:03d}.png"))

        l2 = (torch.tensor(first_frame).float() - baseline_video[0].float()).norm().item()
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
        # Normalise to [0,1] so pairs with different scales are comparable
        l2_arr = np.array(l2_vals, dtype=float)
        l2_norm = (l2_arr - l2_arr.min()) / (l2_arr.max() - l2_arr.min() + 1e-6)
        ax.plot(steps_sorted, l2_norm, "o-", linewidth=2, markersize=5,
                color=color, label=f"[{label}] {desc[:55]}")

    ax.set_xlabel("Step at which source latent was injected")
    ax.set_ylabel("Normalised L2 distance to target baseline\n(0 = identical to target, 1 = maximally distorted)")
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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["1", "2", "both"], default="both",
                        help="1=similarity curve, 2=injection test, both=run all")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Subset of group IDs to run for exp1, e.g. --groups A_attribute E_causal")
    args = parser.parse_args()

    if args.groups:
        for gid in args.groups:
            assert gid in PROMPT_GROUPS, f"Unknown group '{gid}'. Valid: {list(PROMPT_GROUPS)}"
        # Filter to requested groups
        original = PROMPT_GROUPS.copy()
        PROMPT_GROUPS.clear()
        for gid in args.groups:
            PROMPT_GROUPS[gid] = original[gid]

    if args.exp in ("1", "both"):
        run_exp1()
    if args.exp in ("2", "both"):
        run_exp2()

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
