"""
Nirvana-style cross-request latent cache for TeaCache video on Wan 2.1 T2V.

Flow:
1. Encode each incoming prompt with CLIP.
2. Use FAISS to find the nearest cached prompt.
3. If similarity is high enough, choose an intermediate latent tier (k in [5, 10, 15]).
4. Resume Wan denoising from that cached latent for the remaining steps.
5. Re-enable TeaCache only for the resumed suffix of the trajectory.

Important detail:
TeaCache stores step-history-dependent residuals, so those residuals are not
safe to carry across requests together with a cached latent. This implementation
resets TeaCache state whenever a request starts, including cache hits, and then
applies TeaCache only within the remaining denoising steps after the Nirvana
latent is loaded.
"""

import argparse
import importlib.util
import math
import os
import queue
import random
import sys
import time
from contextlib import contextmanager

import faiss
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from eval.teacache.experiments.utils import read_prompt_list
from serving_system_N import KMinHeapCache

WAN_ROOT = os.path.join(os.path.dirname(__file__), "Wan2.1")
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan  # noqa: E402
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from wan.utils.utils import cache_video  # noqa: E402

# Default generation params chosen for Wan 2.1 T2V-1.3B compatibility.
DEFAULT_TASK = "t2v-1.3B"
DEFAULT_SIZE = "832*480"
DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_SAMPLING_STEPS = 50
DEFAULT_GUIDE_SCALE = 5.0
DEFAULT_SAMPLE_SOLVER = "unipc"
DEFAULT_SAMPLE_SHIFT = 5.0
K_VALUES_VIDEO = [2, 7]

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
SIMILARITY_THRESHOLD = 0.85
HIGH_SIMILARITY_THRESHOLD = 0.90


def summarize_prompt(prompt, max_len=72):
    prompt = " ".join(str(prompt).split())
    if len(prompt) <= max_len:
        return prompt
    return prompt[: max_len - 3] + "..."


def log_message(enabled, message):
    if enabled:
        print(message, flush=True)


def rebuild_faiss_index(embeddings, embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)
    if len(embeddings) > 0:
        index.add(embeddings.astype(np.float32, copy=False))
    return index


def remove_cache_entry_from_faiss(cache_id, embedding_dim, index, embeddings, faiss_cache_ids):
    if cache_id not in faiss_cache_ids:
        return index, embeddings, faiss_cache_ids

    remove_pos = faiss_cache_ids.index(cache_id)
    mask = np.ones(len(faiss_cache_ids), dtype=bool)
    mask[remove_pos] = False
    new_embeddings = embeddings[mask]
    new_faiss_cache_ids = [cid for i, cid in enumerate(faiss_cache_ids) if i != remove_pos]
    new_index = rebuild_faiss_index(new_embeddings, embedding_dim)
    return new_index, new_embeddings, new_faiss_cache_ids


def _load_teacache_module():
    module_path = os.path.join(os.path.dirname(__file__), "teacache_wan2.1.py")
    spec = importlib.util.spec_from_file_location("teacache_wan2_1", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEACACHE_WAN = _load_teacache_module()
teacache_forward = TEACACHE_WAN.teacache_forward


def _get_teacache_coefficients(checkpoint_dir, use_ret_steps):
    if use_ret_steps:
        if "1.3B" in checkpoint_dir:
            return (
                [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
                10,
                None,
            )
        if "14B" in checkpoint_dir:
            return (
                [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
                10,
                None,
            )
    else:
        if "1.3B" in checkpoint_dir:
            return (
                [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
                2,
                None,
            )
        if "14B" in checkpoint_dir:
            return (
                [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
                2,
                None,
            )
    raise ValueError(
        "Unable to infer Wan TeaCache coefficients from checkpoint_dir. "
        "Expected path containing '1.3B' or '14B'."
    )


def configure_wan_teacache(model, checkpoint_dir, sampling_steps, teacache_thresh, use_ret_steps):
    model.__class__.enable_teacache = True
    model.__class__.forward = teacache_forward
    model.__class__.teacache_thresh = teacache_thresh
    model.__class__.use_ref_steps = use_ret_steps
    model.__class__.num_steps = sampling_steps * 2
    coefficients, ret_steps, cutoff_steps = _get_teacache_coefficients(
        checkpoint_dir, use_ret_steps
    )
    model.__class__.coefficients = coefficients
    model.__class__.ret_steps = ret_steps
    model.__class__.cutoff_steps = (
        model.__class__.num_steps if cutoff_steps is None else cutoff_steps
    )
    if not use_ret_steps:
        model.__class__.cutoff_steps = max(model.__class__.num_steps - 2, 0)
    reset_wan_teacache_state(model)


def disable_wan_teacache(model):
    model.__class__.enable_teacache = False
    reset_wan_teacache_state(model)


def reset_wan_teacache_state(model):
    model.__class__.cnt = 0
    model.__class__.accumulated_rel_l1_distance_even = 0
    model.__class__.accumulated_rel_l1_distance_odd = 0
    model.__class__.previous_e0_even = None
    model.__class__.previous_e0_odd = None
    model.__class__.previous_residual_even = None
    model.__class__.previous_residual_odd = None
    model.__class__.is_even = True


def normalize_cached_latent(cache_latent):
    if not isinstance(cache_latent, torch.Tensor):
        cache_latent = torch.as_tensor(cache_latent)
    if cache_latent.dim() == 6 and cache_latent.shape[0] == 1:
        cache_latent = cache_latent.squeeze(0)
    if cache_latent.dim() == 5:
        if cache_latent.shape[0] != 1:
            raise ValueError(
                f"Expected cached latent batch size 1, got shape {tuple(cache_latent.shape)}"
            )
        cache_latent = cache_latent.squeeze(0)
    if cache_latent.dim() != 4:
        raise ValueError(f"Invalid cached latent shape: {tuple(cache_latent.shape)}")
    return cache_latent.contiguous()


def get_clip_text_embedding(clip_model, **texts):
    """
    Normalize CLIP text embedding extraction across transformers versions.

    Some environments return the projected text embedding tensor from
    `get_text_features`, while others may surface a model-output object.
    """
    text_features = clip_model.get_text_features(**texts)
    if isinstance(text_features, torch.Tensor):
        return text_features

    if hasattr(text_features, "pooler_output"):
        pooled = text_features.pooler_output
        if hasattr(clip_model, "text_projection"):
            return clip_model.text_projection(pooled)
        return pooled

    if hasattr(text_features, "text_embeds"):
        return text_features.text_embeds

    raise TypeError(
        f"Unsupported CLIP text feature return type: {type(text_features).__name__}"
    )


def wan_generate_with_latent_cache(
    pipeline,
    input_prompt,
    size=(832, 480),
    frame_num=81,
    shift=5.0,
    sample_solver="unipc",
    sampling_steps=50,
    guide_scale=5.0,
    n_prompt="",
    seed=-1,
    offload_model=True,
    collect_latents_at_steps=None,
    cache_latent=None,
    cache_start_step=None,
):
    if cache_latent is not None and cache_start_step is None:
        raise ValueError("cache_start_step must be provided when cache_latent is used")
    if cache_start_step is not None and not (0 <= cache_start_step < sampling_steps):
        raise ValueError(
            f"cache_start_step must be in [0, {sampling_steps - 1}], got {cache_start_step}"
        )

    collect_set = set(collect_latents_at_steps or [])
    collected_latents = []

    F = frame_num
    target_shape = (
        pipeline.vae.model.z_dim,
        (F - 1) // pipeline.vae_stride[0] + 1,
        size[1] // pipeline.vae_stride[1],
        size[0] // pipeline.vae_stride[2],
    )

    seq_len = (
        math.ceil(
            (target_shape[2] * target_shape[3])
            / (pipeline.patch_size[1] * pipeline.patch_size[2])
            * target_shape[1]
            / pipeline.sp_size
        )
        * pipeline.sp_size
    )

    if n_prompt == "":
        n_prompt = pipeline.sample_neg_prompt

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=pipeline.device)
    seed_g.manual_seed(seed)

    if not pipeline.t5_cpu:
        pipeline.text_encoder.model.to(pipeline.device)
        context = pipeline.text_encoder([input_prompt], pipeline.device)
        context_null = pipeline.text_encoder([n_prompt], pipeline.device)
        if offload_model:
            pipeline.text_encoder.model.cpu()
    else:
        context = pipeline.text_encoder([input_prompt], torch.device("cpu"))
        context_null = pipeline.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(pipeline.device) for t in context]
        context_null = [t.to(pipeline.device) for t in context_null]

    if cache_latent is None:
        latents = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=pipeline.device,
                generator=seed_g,
            )
        ]
        start_step = 0
    else:
        latents = [normalize_cached_latent(cache_latent).to(pipeline.device, dtype=torch.float32)]
        start_step = cache_start_step

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=pipeline.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipeline.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=pipeline.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipeline.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=pipeline.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError("Unsupported solver.")

        active_timesteps = timesteps[start_step:]
        remaining_steps = len(active_timesteps)
        if remaining_steps == 0:
            raise ValueError("No diffusion steps remain after cache_start_step")

        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        pipeline.model.to(pipeline.device)
        x0 = None
        for local_idx, t in enumerate(active_timesteps):
            global_step = start_step + local_idx + 1
            timestep = torch.stack([t]).to(pipeline.device)

            noise_pred_cond = pipeline.model(latents, t=timestep, **arg_c)[0]
            noise_pred_uncond = pipeline.model(latents, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

            if global_step in collect_set:
                collected_latents.append(latents[0].detach().cpu().clone())

            x0 = latents

        if offload_model:
            pipeline.model.cpu()
            torch.cuda.empty_cache()
        if pipeline.rank == 0:
            videos = pipeline.vae.decode(x0)

    del latents
    del sample_scheduler
    if offload_model:
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()

    if collect_latents_at_steps is not None:
        return videos[0] if pipeline.rank == 0 else None, collected_latents
    return videos[0] if pipeline.rank == 0 else None


def request_scheduler_video(
    req_queue,
    selected_requests,
    start_time,
    index,
    embedding_dim,
    cache,
    new_cache_queue,
    cached_requests,
    faiss_cache_ids,
    final_text_embeddings,
    k_values,
    worker_status,
    clip_model_id,
    num_workers,
    log_enabled=False,
    log_file="request_throughput_video_teacache.csv",
    eval_mode=False,
    no_nirvana=False,
    cache_stats=None,
):
    device = "cpu"
    processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
    agg_k_distribution = {k: 0 for k in k_values}

    if cache_stats is None:
        cache_stats = {"hits": 0, "misses": 0}

    minute = 0
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w") as f:
        f.write("timestamp,request_rate,throughput\n")
    last_check_time_queue = time.time()
    request_count_per_min = 0
    size_of_queues = 0

    for _, row in selected_requests.iterrows():
        if not eval_mode:
            while time.time() - start_time < row["seconds_from_start"]:
                time.sleep(0.1)

        request_arrival_time = time.time()
        row["start_time"] = request_arrival_time

        # 1. ENCODING & VECTOR SEARCH TIME
        search_start = time.perf_counter()

        while not new_cache_queue.empty():
            cache_data = new_cache_queue.get()
            new_cached_latents = [z.clone() for z in cache_data["cached_latents"]]
            new_cached_prompt = cache_data["prompt"]
            new_query_embedding = cache_data["query_embedding"]
            new_cache_id = cache_data["cache_id"]

            while len(cache.item_map) + len(cache.k_values) > cache.max_size:
                evicted_cache_id = cache.evict()
                if evicted_cache_id is not None:
                    cached_requests.pop(evicted_cache_id, None)
                    index, final_text_embeddings, faiss_cache_ids = remove_cache_entry_from_faiss(
                        evicted_cache_id,
                        embedding_dim,
                        index,
                        final_text_embeddings,
                        faiss_cache_ids,
                    )

            cached_requests[new_cache_id] = new_cached_prompt
            faiss_cache_ids.append(new_cache_id)
            index.add(new_query_embedding)
            final_text_embeddings = np.concatenate(
                (final_text_embeddings, new_query_embedding), axis=0
            )
            for idx, k in enumerate(k_values):
                cache.insert(new_cache_id, 0, k, new_cached_latents[idx])

        prompt = row["prompt"]
        request_id = row["request_id"]
        texts = processor(
            text=[prompt],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=77,
        ).to(device)
        with torch.no_grad():
            text_embedding = get_clip_text_embedding(clip_model, **texts).cpu()

        if no_nirvana:
            row["cached"] = None
            row["k"] = None
            row["latent"] = None
            row["query_embedding"] = text_embedding.clone()
            log_message(
                log_enabled,
                f"[Scheduler] request={request_id} mode=miss reason=no_nirvana "
                f"prompt='{summarize_prompt(prompt)}'",
            )
            cache_stats["misses"] = cache_stats.get("misses", 0) + 1
            req_queue.put(row.to_dict())
            continue

        query_embedding = text_embedding.numpy().reshape(1, -1).astype(np.float32)

        if index.ntotal == 0:
            row["cached"] = None
            row["k"] = None
            row["latent"] = None
            row["query_embedding"] = text_embedding.clone()
            log_message(
                log_enabled,
                f"[Scheduler] request={request_id} mode=miss reason=empty_cache "
                f"prompt='{summarize_prompt(prompt)}'",
            )
            cache_stats["misses"] = cache_stats.get("misses", 0) + 1
            req_queue.put(row.to_dict())
        else:
            distances, indices = index.search(query_embedding, k=1)
            closest_faiss_pos = indices[0][0]
            closest_cache_id = faiss_cache_ids[closest_faiss_pos]
            closest_prompt = cached_requests[closest_cache_id]
            closest_prompt_summary = summarize_prompt(closest_prompt)
            closest_texts = processor(
                text=[closest_prompt],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=77,
            ).to(device)
            with torch.no_grad():
                closest_text_embedding = get_clip_text_embedding(
                    clip_model, **closest_texts
                )

            text_embedding_device = text_embedding.to(device)
            with torch.no_grad():
                text_norm = text_embedding_device / text_embedding_device.norm(
                    dim=-1, keepdim=True
                )
                closest_text_norm = closest_text_embedding / closest_text_embedding.norm(
                    dim=-1, keepdim=True
                )
                text_similarity_scores = torch.matmul(text_norm, closest_text_norm.T)
            text_similarity_scores = torch.clamp(text_similarity_scores, min=0)
            text_embedding = text_embedding_device.cpu()
            similarity = text_similarity_scores.item()
            search_end = time.perf_counter()
            row["vector_search_ms"] = (search_end - search_start) * 1000

            if similarity > SIMILARITY_THRESHOLD:
                if similarity > HIGH_SIMILARITY_THRESHOLD:
                    closest_index = 7
                else:
                    closest_index = 2

                retrieval_start = time.perf_counter()
                best_candidate = cache.retrieve(closest_index, closest_cache_id)
                retrieval_end = time.perf_counter()
                row["cache_retrieval_ms"] = (retrieval_end - retrieval_start) * 1000

                if best_candidate:
                    _, (_, k_i, latent) = best_candidate
                    row["cached"] = True
                    row["k"] = k_i
                    row["latent"] = latent.clone().to(dtype=torch.float32).cpu()
                    row["query_embedding"] = text_embedding.clone()
                    log_message(
                        log_enabled,
                        f"[Scheduler] request={request_id} mode=hit sim={similarity:.3f} "
                        f"k={k_i} prompt='{summarize_prompt(prompt)}' "
                        f"nearest_prompt='{closest_prompt_summary}'",
                    )
                    cache_stats["hits"] = cache_stats.get("hits", 0) + 1
                    req_queue.put(row.to_dict())
                    agg_k_distribution[k_i] += 1
                else:
                    row["cached"] = None
                    row["k"] = None
                    row["latent"] = None
                    row["query_embedding"] = text_embedding.clone()
                    log_message(
                        log_enabled,
                        f"[Scheduler] request={request_id} mode=miss reason=cache_lookup_failed "
                        f"sim={similarity:.3f} prompt='{summarize_prompt(prompt)}' "
                        f"nearest_prompt='{closest_prompt_summary}'",
                    )
                    cache_stats["misses"] = cache_stats.get("misses", 0) + 1
                    req_queue.put(row.to_dict())
            else:
                row["cached"] = None
                row["k"] = None
                row["latent"] = None
                row["query_embedding"] = text_embedding.clone()
                log_message(
                    log_enabled,
                    f"[Scheduler] request={request_id} mode=miss reason=low_similarity "
                    f"sim={similarity:.3f} prompt='{summarize_prompt(prompt)}' "
                    f"nearest_prompt='{closest_prompt_summary}'",
                )
                cache_stats["misses"] = cache_stats.get("misses", 0) + 1
                req_queue.put(row.to_dict())

        row["queue_entry_time"] = time.time()
        
        request_count_per_min += 1
        current_time = time.time()
        if current_time - last_check_time_queue >= 60:
            elapsed_time = current_time - last_check_time_queue
            minute += 1
            new_size_of_queues = req_queue.qsize()
            throughput = size_of_queues + request_count_per_min - new_size_of_queues
            size_of_queues = new_size_of_queues
            with open(log_file, "a") as f:
                f.write(
                    f"{minute},{request_count_per_min / elapsed_time * 60},{throughput / elapsed_time * 60}\n"
                )
            request_count_per_min = 0
            last_check_time_queue = current_time

    if eval_mode:
        for _ in range(num_workers):
            req_queue.put(None)
        log_message(log_enabled, "[Scheduler] eval_mode complete, sent shutdown sentinels")
    else:
        for _ in range(num_workers):
            req_queue.put(None)
        log_message(log_enabled, "[Scheduler] request list complete, sent shutdown sentinels")

    while True:
        all_done = all(status in ["finished", "dropped"] for status in worker_status.values())
        if all_done:
            break
        time.sleep(1)


def worker_video(
    gpu_id,
    req_queue,
    new_cache_queue,
    latency_queue,
    worker_status,
    video_directory,
    ckpt_dir,
    task,
    size_name,
    num_frames,
    sampling_steps,
    sample_solver,
    sample_shift,
    guide_scale,
    teacache_thresh,
    use_ret_steps,
    offload_model,
    disable_teacache,
    log_enabled,
    eval_mode,
    loop=1,
    no_nirvana=False,
):
    torch.cuda.set_device(gpu_id)

    cfg = WAN_CONFIGS[task]
    video_size = SIZE_CONFIGS[size_name]
    pipeline = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=gpu_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    pipeline.__class__.generate = wan_generate_with_latent_cache
    if disable_teacache:
        disable_wan_teacache(pipeline.model)
    else:
        configure_wan_teacache(
            pipeline.model,
            checkpoint_dir=ckpt_dir,
            sampling_steps=sampling_steps,
            teacache_thresh=teacache_thresh,
            use_ret_steps=use_ret_steps,
        )
    worker_status[gpu_id] = "running"

    idle_counter = 0
    max_idle_iterations = 1 if eval_mode else 100

    while True:
        try:
            request = req_queue.get(timeout=10)
            if request is None:
                worker_status[gpu_id] = "finished"
                log_message(log_enabled, f"[Worker {gpu_id}] received shutdown sentinel")
                break
            process_start = time.time()
            idle_counter = 0
            prompt = request["prompt"]
            request_id = request.get("request_id", "unknown")
            cache_mode = "hit" if request["cached"] else "miss"
            generation_total_elapsed = 0.0
            log_message(
                log_enabled,
                f"[Worker {gpu_id}] request={request_id} start mode={cache_mode} "
                f"prompt='{summarize_prompt(prompt)}'",
            )

            for l in range(loop):
                out_path = os.path.join(video_directory, f"{prompt}-{l}.mp4")
                generation_start = time.time()
                log_message(
                    log_enabled,
                    f"[Worker {gpu_id}] request={request_id} generation={l + 1}/{loop} "
                    f"status=started mode={cache_mode}",
                )

                if request["cached"] is None:
                    if disable_teacache:
                        disable_wan_teacache(pipeline.model)
                    else:
                        reset_wan_teacache_state(pipeline.model)
                        configure_wan_teacache(
                            pipeline.model,
                            checkpoint_dir=ckpt_dir,
                            sampling_steps=sampling_steps,
                            teacache_thresh=teacache_thresh,
                            use_ret_steps=use_ret_steps,
                        )
                    collect_latents = tuple(K_VALUES_VIDEO) if l == 0 else None
                    result = pipeline.generate(
                        prompt,
                        size=video_size,
                        frame_num=num_frames,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=l,
                        offload_model=offload_model,
                        collect_latents_at_steps=collect_latents,
                    )
                    if isinstance(result, tuple):
                        video, collected_latents = result
                    else:
                        video = result
                        collected_latents = None

                    cache_video(
                        tensor=video[None],
                        save_file=out_path,
                        fps=cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )

                    if (
                        not no_nirvana
                        and collected_latents is not None
                        and request.get("query_embedding") is not None
                    ):
                        cached_latents = [z.cpu().clone() for z in collected_latents]
                        qe = request["query_embedding"]
                        qe_np = (
                            qe.numpy().reshape(1, -1)
                            if hasattr(qe, "numpy")
                            else np.array(qe).reshape(1, -1)
                        ).astype(np.float32)
                        new_cache_queue.put(
                            {
                                "cache_id": request_id,
                                "cached_latents": cached_latents,
                                "prompt": prompt,
                                "query_embedding": qe_np,
                            }
                        )
                else:
                    cache_latent = normalize_cached_latent(request["latent"])
                    k = request["k"]
                    remaining_steps = sampling_steps - k
                    if disable_teacache:
                        disable_wan_teacache(pipeline.model)
                    else:
                        configure_wan_teacache(
                            pipeline.model,
                            checkpoint_dir=ckpt_dir,
                            sampling_steps=remaining_steps,
                            teacache_thresh=teacache_thresh,
                            use_ret_steps=use_ret_steps,
                        )
                    result = pipeline.generate(
                        prompt,
                        size=video_size,
                        frame_num=num_frames,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=l,
                        offload_model=offload_model,
                        cache_latent=cache_latent,
                        cache_start_step=k,
                    )
                    video = result[0] if isinstance(result, tuple) else result
                    cache_video(
                        tensor=video[None],
                        save_file=out_path,
                        fps=cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )

                generation_elapsed = time.time() - generation_start
                generation_total_elapsed += generation_elapsed
                log_message(
                    log_enabled,
                    f"[Worker {gpu_id}] request={request_id} generation={l + 1}/{loop} "
                    f"status=finished elapsed={generation_elapsed:.2f}s output='{out_path}'",
                )

            finish_time = time.time() - request["start_time"]
            pure_processing_time = time.time() - process_start
            log_message(
                log_enabled,
                f"[Worker {gpu_id}] request={request_id} completed "
                f"queue_to_finish={finish_time:.2f}s processing={pure_processing_time:.2f}s "
                f"generation={generation_total_elapsed:.2f}s",
            )
            latency_queue.put((
                finish_time,
                pure_processing_time,
                cache_mode,
                generation_total_elapsed,
                float(request.get("vector_search_ms") or 0),
                float(request.get("cache_retrieval_ms") or 0),
            ))
        except queue.Empty:
            idle_counter += 1
            if idle_counter >= max_idle_iterations:
                worker_status[gpu_id] = "dropped"
                break
            continue
        except Exception:
            worker_status[gpu_id] = "dropped"
            raise


def generate_rapidly_increasing_seconds_from_start(
    num_requests, min_rate=2, max_rate=9, duration=100 * 60
):
    del duration
    min_rate_per_sec = min_rate / 60
    max_rate_per_sec = max_rate / 60
    x = np.linspace(-2, 6, num_requests)
    sigmoid_growth = 1 / (1 + np.exp(-1.5 * x))
    request_rates = min_rate_per_sec + (max_rate_per_sec - min_rate_per_sec) * sigmoid_growth
    interarrival_times = 1 / np.maximum(request_rates, 1e-3)
    return np.cumsum(interarrival_times)


def generate_fixed_interval_seconds_from_start(num_requests, interval_seconds):
    if num_requests <= 0:
        return np.array([], dtype=np.float32)
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be non-negative")
    return np.arange(num_requests, dtype=np.float32) * float(interval_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="TeaCache video serving with Nirvana-style cross-request latent cache on Wan 2.1"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to Wan 2.1 checkpoint directory")
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        choices=["t2v-1.3B", "t2v-14B"],
        help="Wan 2.1 text-to-video base model",
    )
    parser.add_argument(
        "--num_req",
        type=int,
        default=None,
        help="Max prompts to process (default: all when --prompt_list given, else 50)",
    )
    parser.add_argument("--cache_size", type=int, default=1000, help="cache size (requests)")
    parser.add_argument(
        "--video_directory",
        type=str,
        default="./video_outputs_teacache",
        help="directory for generated videos",
    )
    parser.add_argument(
        "--prompt_list",
        type=str,
        default=None,
        help="JSON file with list of prompts; each item can have 'prompt_en'",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=DEFAULT_SIZE,
        choices=sorted(SIZE_CONFIGS.keys()),
        help="Wan video size, e.g. 832*480",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="number of frames; Wan expects 4n+1",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=DEFAULT_NUM_SAMPLING_STEPS,
        help="total Wan diffusion steps",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default=DEFAULT_SAMPLE_SOLVER,
        choices=["unipc", "dpm++"],
        help="Wan sampler",
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=DEFAULT_SAMPLE_SHIFT,
        help="Wan sampling shift",
    )
    parser.add_argument(
        "--guide_scale",
        type=float,
        default=DEFAULT_GUIDE_SCALE,
        help="classifier-free guidance scale",
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="TeaCache rel_l1 threshold; higher means more skipping",
    )
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Enable Wan TeaCache retention-step configuration",
    )
    parser.add_argument(
        "--no_teacache",
        action="store_true",
        help="Disable TeaCache and run plain Wan denoising",
    )
    parser.add_argument(
        "--offload_model",
        action="store_true",
        help="Offload Wan submodules between steps to save VRAM",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=5,
        help="Videos per prompt; only the first miss writes Nirvana cache",
    )
    parser.add_argument(
        "--eval_mode",
        action="store_true",
        help="No request timing: submit all prompts at once and run as fast as possible",
    )
    parser.add_argument(
        "--request_interval_seconds",
        type=float,
        default=None,
        help="In non-eval mode, use a fixed gap between request arrivals instead of the default rising-rate schedule",
    )
    parser.add_argument(
        "--no_nirvana",
        action="store_true",
        help="Disable Nirvana cache: every request does full generation",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Print per-request scheduler and worker progress logs",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="request_throughput_video_teacache_w_nirvana.csv",
        help="log file path",
    )
    args = parser.parse_args()

    if args.size not in SUPPORTED_SIZES[args.task]:
        raise ValueError(
            f"Unsupported size {args.size} for {args.task}; choose from {SUPPORTED_SIZES[args.task]}"
        )
    if args.sample_steps <= max(K_VALUES_VIDEO):
        raise ValueError(
            f"--sample_steps must be greater than max cache step {max(K_VALUES_VIDEO)}"
        )

    os.makedirs(args.video_directory, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices")

    if args.prompt_list and os.path.isfile(args.prompt_list):
        prompts = read_prompt_list(args.prompt_list)
        if args.num_req is not None:
            prompts = prompts[: args.num_req]
    else:
        num_req = args.num_req if args.num_req is not None else 50
        prompts = [
            "A cat walking on the street.",
            "Ocean waves under sunset.",
            "A dog running in the park.",
        ] * max(1, (num_req + 2) // 3)
        prompts = prompts[:num_req]

    if args.request_interval_seconds is not None:
        seconds_from_start = generate_fixed_interval_seconds_from_start(
            len(prompts), args.request_interval_seconds
        )
    else:
        seconds_from_start = generate_rapidly_increasing_seconds_from_start(
            len(prompts), min_rate=0.5, max_rate=4
        )
    selected_requests = pd.DataFrame(
        {
            "request_id": list(range(len(prompts))),
            "prompt": prompts,
            "seconds_from_start": seconds_from_start,
        }
    )

    embedding_dim = 768
    index = faiss.IndexFlatL2(embedding_dim)
    final_text_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
    cached_requests = {}
    faiss_cache_ids = []
    cache = KMinHeapCache(
        max_size=args.cache_size * len(K_VALUES_VIDEO),
        initial_embeddings=final_text_embeddings,
        latents=torch.empty(0),
        k_values=K_VALUES_VIDEO,
    )

    req_queue = mp.Queue()
    new_cache_queue = mp.Queue()
    latency_queue = mp.Queue()
    manager = mp.Manager()
    worker_status = manager.dict()
    cache_stats = manager.dict()
    cache_stats["hits"] = 0
    cache_stats["misses"] = 0

    wall_start = time.time()
    scheduler = mp.Process(
        target=request_scheduler_video,
        args=(
            req_queue,
            selected_requests,
            wall_start,
            index,
            embedding_dim,
            cache,
            new_cache_queue,
            cached_requests,
            faiss_cache_ids,
            final_text_embeddings,
            K_VALUES_VIDEO,
            worker_status,
            CLIP_MODEL_ID,
            num_gpus,
        ),
        kwargs={
            "log_enabled": args.log,
            "log_file": args.log_file,
            "eval_mode": args.eval_mode,
            "no_nirvana": args.no_nirvana,
            "cache_stats": cache_stats,
        },
    )
    scheduler.start()

    workers = []
    for gpu_id in range(num_gpus):
        worker_status[gpu_id] = "starting"
        p = mp.Process(
            target=worker_video,
            args=(
                gpu_id,
                req_queue,
                new_cache_queue,
                latency_queue,
                worker_status,
                args.video_directory,
                args.ckpt_dir,
                args.task,
                args.size,
                args.num_frames,
                args.sample_steps,
                args.sample_solver,
                args.sample_shift,
                args.guide_scale,
                args.teacache_thresh,
                args.use_ret_steps,
                args.offload_model,
                args.no_teacache,
                args.log,
                args.eval_mode,
                args.loop,
                args.no_nirvana,
            ),
        )
        p.start()
        workers.append(p)

    total_requests = len(prompts)
    all_latencies = []
    all_processing_times = []
    hit_processing_times = []
    miss_processing_times = []
    hit_generation_times = []
    miss_generation_times = []
    all_vector_search_ms = []
    all_cache_retrieval_ms = []
    with tqdm(total=total_requests, desc="Requests", unit="req") as pbar:
        for _ in range(total_requests):
            finish_time, pure_processing_time, cache_mode, gen_s, vsearch_ms, retrieval_ms = latency_queue.get()
            all_latencies.append(finish_time)
            all_processing_times.append(pure_processing_time)
            if cache_mode == "hit":
                hit_processing_times.append(pure_processing_time)
                hit_generation_times.append(gen_s)
            else:
                miss_processing_times.append(pure_processing_time)
                miss_generation_times.append(gen_s)
            if vsearch_ms > 0:
                all_vector_search_ms.append(vsearch_ms)
            if retrieval_ms > 0:
                all_cache_retrieval_ms.append(retrieval_ms)
            pbar.update(1)

    for p in workers:
        p.join()
    scheduler.join()

    wall_total = time.time() - wall_start
    print(f"[Total wall time] {wall_total:.2f}s")
    if all_latencies:
        print(
            f"[Per-request latency] min={min(all_latencies):.2f}s max={max(all_latencies):.2f}s avg={np.mean(all_latencies):.2f}s (n={len(all_latencies)})"
        )
    if all_processing_times:
        print(
            f"[Pure processing time] min={min(all_processing_times):.2f}s max={max(all_processing_times):.2f}s avg={np.mean(all_processing_times):.2f}s (n={len(all_processing_times)})"
        )
    if hit_processing_times:
        print(
            f"[Hit processing time] min={min(hit_processing_times):.2f}s max={max(hit_processing_times):.2f}s avg={np.mean(hit_processing_times):.2f}s (n={len(hit_processing_times)})"
        )
    if miss_processing_times:
        print(
            f"[Miss processing time] min={min(miss_processing_times):.2f}s max={max(miss_processing_times):.2f}s avg={np.mean(miss_processing_times):.2f}s (n={len(miss_processing_times)})"
        )
    if hit_generation_times:
        print(
            f"[Hit generation time] min={min(hit_generation_times):.2f}s max={max(hit_generation_times):.2f}s avg={np.mean(hit_generation_times):.2f}s (n={len(hit_generation_times)})"
        )
    if miss_generation_times:
        print(
            f"[Miss generation time] min={min(miss_generation_times):.2f}s max={max(miss_generation_times):.2f}s avg={np.mean(miss_generation_times):.2f}s (n={len(miss_generation_times)})"
        )
    if all_vector_search_ms:
        print(
            f"[Vector search time] min={min(all_vector_search_ms):.2f}ms max={max(all_vector_search_ms):.2f}ms avg={np.mean(all_vector_search_ms):.2f}ms (n={len(all_vector_search_ms)})"
        )
    if all_cache_retrieval_ms:
        print(
            f"[Cache retrieval time] min={min(all_cache_retrieval_ms):.2f}ms max={max(all_cache_retrieval_ms):.2f}ms avg={np.mean(all_cache_retrieval_ms):.2f}ms (n={len(all_cache_retrieval_ms)})"
        )
    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    total_cache_requests = hits + misses
    if total_cache_requests > 0:
        hit_rate_pct = 100.0 * hits / total_cache_requests
        print(f"[Cache hit rate] {hits}/{total_cache_requests} = {hit_rate_pct:.1f}%")
    elif args.no_nirvana:
        print("[Cache hit rate] N/A (Nirvana disabled)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
