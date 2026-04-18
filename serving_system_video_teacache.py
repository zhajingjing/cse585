"""
Nirvana-style cross-request latent cache for TeaCache video (Open-Sora).
Uses k_values = [5, 10, 15]; CLIP for prompt similarity; FAISS + KMinHeapCache.
"""
import os
import time
import queue
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pandas as pd
import faiss
from transformers import CLIPModel, CLIPProcessor

from serving_system_N import evict_from_faiss, KMinHeapCache

# Video pipeline + TeaCache (same as eval/teacache/experiments/opensora.py)
from videosys import OpenSoraConfig, VideoSysEngine
from eval.teacache.experiments.opensora import teacache_forward
from eval.teacache.experiments.utils import read_prompt_list

# Default generation params (fixed for cache compatibility)
DEFAULT_RESOLUTION = "480p"
DEFAULT_ASPECT_RATIO = "9:16"
DEFAULT_NUM_FRAMES = 51
K_VALUES_VIDEO = [5, 10, 15]


def _drain_new_cache_queue(
    new_cache_queue, cache, cached_requests, index, final_text_embeddings, k_values
):
    """Drain pending latent-cache entries from workers into the cache.

    Normalizes embeddings for IndexFlatIP, evicts if needed, and remaps
    KMinHeapCache indices to stay aligned with the FAISS index after eviction.

    Returns updated (index, final_text_embeddings).
    """
    while not new_cache_queue.empty():
        cache_data = new_cache_queue.get()
        new_cached_latents = [z.clone() for z in cache_data["cached_latents"]]
        new_cached_prompt = cache_data["prompt"]
        new_query_embedding = cache_data["query_embedding"]

        while len(cache.item_map) + len(cache.k_values) > cache.max_size:
            evicted_index = cache.evict()
            if evicted_index is not None:
                del cached_requests[evicted_index]
                index, final_text_embeddings = evict_from_faiss(
                    index, final_text_embeddings, evicted_index, use_ip=True
                )
                cache.remap_index_after_eviction(evicted_index)

        # Normalize before inserting into IndexFlatIP so inner-product == cosine.
        normalized_embedding = new_query_embedding.copy()
        faiss.normalize_L2(normalized_embedding)

        num_embeddings = index.ntotal
        cached_requests.append(new_cached_prompt)
        index.add(normalized_embedding)
        final_text_embeddings = np.concatenate(
            (final_text_embeddings, normalized_embedding), axis=0
        )
        for idx, k in enumerate(k_values):
            cache.insert(num_embeddings, 0, k, new_cached_latents[idx])

    return index, final_text_embeddings

def request_scheduler_video(
    req_queue,
    selected_requests,
    start_time,
    index,
    cache,
    new_cache_queue,
    cached_requests,
    final_text_embeddings,
    k_values,
    clip_model_name,
    clip_device,
    worker_status,
    done_event,
    log_file="request_throughput_video_teacache.csv",
    eval_mode=False,
    no_nirvana=False,
    cache_stats=None,
):
    # Load CLIP inside the subprocess so it owns the CUDA tensors and can
    # release them cleanly on exit (avoids CudaIPCTypes producer warning).
    processor  = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(clip_device)
    device = clip_device
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

        # Bug 2 fix: use helper that also calls remap_index_after_eviction
        # Bug 4 fix: helper normalizes embeddings for IndexFlatIP
        index, final_text_embeddings = _drain_new_cache_queue(
            new_cache_queue, cache, cached_requests, index, final_text_embeddings, k_values
        )

        prompt = row["prompt"]
        if no_nirvana:
            with torch.no_grad():
                texts = processor(
                    text=[prompt],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=77,
                ).to(device)
                text_embedding = clip_model.get_text_features(**texts).cpu()
            row["cached"] = None
            row["k"] = None
            row["latent"] = None
            row["query_embedding"] = text_embedding.clone()
            cache_stats["misses"] = cache_stats.get("misses", 0) + 1
            req_queue.put(row.to_dict())
            continue

        texts = processor(
            text=[prompt],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=77,
        ).to(device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**texts).cpu()

        # Bug 4 fix: normalize query before searching IndexFlatIP;
        # inner-product on unit vectors == cosine similarity, eliminating the
        # redundant second CLIP call that was needed with IndexFlatL2.
        query_embedding = text_embedding.numpy().reshape(1, -1).copy()
        faiss.normalize_L2(query_embedding)

        if index.ntotal == 0:
            row["cached"] = None
            row["k"] = None
            row["latent"] = None
            row["query_embedding"] = text_embedding.clone()
            cache_stats["misses"] = cache_stats.get("misses", 0) + 1
            req_queue.put(row.to_dict())
        else:
            distances, indices = index.search(query_embedding, k=1)
            # With IndexFlatIP + L2-normalized vectors the distance IS cosine similarity.
            text_similarity = float(distances[0][0])

            if text_similarity > 0.65:
                # Opt 2: finer-grained k mapping across the full (0.65, 1.0] range
                if text_similarity > 0.95:
                    closest_index = 15
                elif text_similarity > 0.85:
                    closest_index = 10
                elif text_similarity > 0.75:
                    closest_index = 10
                else:
                    closest_index = 5
                best_candidate = cache.retrieve(closest_index, indices[0][0])
                if best_candidate:
                    score, (idex, k_i, latent) = best_candidate
                    row["cached"] = True
                    row["k"] = k_i
                    row["latent"] = latent.clone().to(dtype=torch.float32).cpu()
                    row["query_embedding"] = text_embedding.clone()
                    cache_stats["hits"] = cache_stats.get("hits", 0) + 1
                    req_queue.put(row.to_dict())
                    agg_k_distribution[k_i] += 1
                else:
                    row["cached"] = None
                    row["k"] = None
                    row["latent"] = None
                    row["query_embedding"] = text_embedding.clone()
                    cache_stats["misses"] = cache_stats.get("misses", 0) + 1
                    req_queue.put(row.to_dict())
            else:
                row["cached"] = None
                row["k"] = None
                row["latent"] = None
                row["query_embedding"] = text_embedding.clone()
                cache_stats["misses"] = cache_stats.get("misses", 0) + 1
                req_queue.put(row.to_dict())

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

    # Drain any remaining cache updates while workers finish their last requests.
    # Bug D fix: sleep to avoid busy-spinning when both queues are temporarily empty.
    while not req_queue.empty():
        index, final_text_embeddings = _drain_new_cache_queue(
            new_cache_queue, cache, cached_requests, index, final_text_embeddings, k_values
        )
        time.sleep(0.05)

    # Bug F fix: log k distribution so cache skipping behaviour is observable.
    print(f"[Scheduler] k distribution: {agg_k_distribution}")

    # Bug 3 fix: signal workers that no more requests are coming so they can
    # exit cleanly as "finished" rather than waiting ~1000 s for idle timeout.
    done_event.set()

    while True:
        if req_queue.empty():
            all_done = all(
                status in ["finished", "dropped"] for status in worker_status.values()
            )
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
    resolution,
    aspect_ratio,
    num_frames,
    teacache_thresh,
    done_event,
    loop=1,
    no_nirvana=False,
):
    device = f"cuda:{gpu_id}"
    config = OpenSoraConfig(num_gpus=1, num_sampling_steps=30)
    engine = VideoSysEngine(config)

    # Bug 1 fix: set TeaCache state on the instance, not the class, so multiple
    # workers don't overwrite each other's accumulated_rel_l1_distance etc.
    trans = engine.driver_worker.transformer
    trans.enable_teacache = True
    trans.rel_l1_thresh = teacache_thresh
    trans.accumulated_rel_l1_distance = 0
    trans.previous_modulated_input = None
    trans.previous_residual = None
    trans.__class__.forward = teacache_forward  # method binding must stay on class

    pipeline = engine.driver_worker
    idle_counter = 0
    max_idle_iterations = 100

    while True:
        try:
            request = req_queue.get(timeout=10)
            process_start = time.time()
            idle_counter = 0
            prompt = request["prompt"]
            # Same naming as eval/teacache/experiments/utils.py: {prompt}-{loop_idx}.mp4
            for loop_idx in range(loop):  # Opt 3: renamed l → loop_idx (l/1 ambiguity)
                out_path = os.path.join(video_directory, f"{prompt}-{loop_idx}.mp4")

                if request["cached"] is None:
                    # Full generation (cache miss); collect latents only on first iteration
                    collect_latents = tuple(K_VALUES_VIDEO) if (loop_idx == 0) else None
                    result = pipeline.generate(
                        prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        num_frames=num_frames,
                        seed=loop_idx,
                        verbose=False,
                        collect_latents_at_steps=collect_latents,
                    )
                    if isinstance(result, tuple):
                        output, collected_latents = result
                    else:
                        output = result
                        collected_latents = None
                    video = output.video[0]
                    engine.save_video(video, out_path)
                    if not no_nirvana and collected_latents is not None and request.get("query_embedding") is not None:
                        cached_latents = [z.cpu().clone() for z in collected_latents]
                        qe = request["query_embedding"]
                        qe_np = qe.numpy().reshape(1, -1) if hasattr(qe, "numpy") else np.array(qe).reshape(1, -1)
                        new_cache_queue.put(
                            {
                                "cached_latents": cached_latents,
                                "prompt": prompt,
                                "query_embedding": qe_np,
                            }
                        )
                else:
                    # Nirvana hit: use cached latent, only seed differs per loop iteration
                    cache_latent = request["latent"]
                    # Normalize cached latent shape to [B, C, T, H, W].
                    # Stored latents may already include batch dim (5D), older entries can be 6D.
                    if isinstance(cache_latent, torch.Tensor):
                        if cache_latent.dim() == 4:
                            cache_latent = cache_latent.unsqueeze(0)
                        elif cache_latent.dim() == 6 and cache_latent.shape[0] == 1:
                            cache_latent = cache_latent.squeeze(0)
                        if cache_latent.dim() != 5:
                            raise ValueError(f"Invalid cached latent shape: {tuple(cache_latent.shape)}")
                    k = request["k"]
                    result = pipeline.generate(
                        prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        num_frames=num_frames,
                        seed=loop_idx,
                        verbose=False,
                        cache_latent=cache_latent,
                        cache_start_step=k,
                    )
                    if isinstance(result, tuple):
                        output = result[0]
                    else:
                        output = result
                    video = output.video[0]
                    engine.save_video(video, out_path)

            finish_time = time.time() - request["start_time"]
            pure_processing_time = time.time() - process_start
            latency_queue.put((finish_time, pure_processing_time))

        except queue.Empty:
            # Bug 3 fix: exit cleanly as "finished" once the scheduler signals
            # no more requests, rather than waiting ~1000 s for idle timeout.
            if done_event.is_set():
                worker_status[gpu_id] = "finished"
                break
            idle_counter += 1
            if idle_counter >= max_idle_iterations:
                worker_status[gpu_id] = "dropped"
                break
            continue


def generate_rapidly_increasing_seconds_from_start(
    num_requests, min_rate=2, max_rate=9, duration=100 * 60
):
    min_rate_per_sec = min_rate / 60
    max_rate_per_sec = max_rate / 60
    x = np.linspace(-2, 6, num_requests)
    sigmoid_growth = 1 / (1 + np.exp(-1.5 * x))
    request_rates = min_rate_per_sec + (max_rate_per_sec - min_rate_per_sec) * sigmoid_growth
    interarrival_times = 1 / np.maximum(request_rates, 1e-3)
    return np.cumsum(interarrival_times)


def main():
    parser = argparse.ArgumentParser(
        description="TeaCache video serving with Nirvana-style cross-request latent cache"
    )
    parser.add_argument(
        "--num_req",
        type=int,
        default=None,
        help="Max prompts to process (default: all when --prompt_list given, else 50)",
    )
    parser.add_argument("--cache_size", type=int, default=1000, help="cache size (entries)")
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
        help="JSON file with list of prompts (e.g. VBench_full_info.json); each item can have 'prompt_en'",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=DEFAULT_RESOLUTION,
        help="resolution (e.g. 480p)",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default=DEFAULT_ASPECT_RATIO,
        help="aspect ratio (e.g. 9:16)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="number of frames",
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="TeaCache rel_l1_thresh (same as opensora teacache_fast=0.2; higher = more skip)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=5,
        help="Videos per prompt (same as opensora eval, default 5); only the first uses Nirvana cache",
    )
    parser.add_argument(
        "--eval_mode",
        action="store_true",
        help="No request timing: submit all prompts at once and run as fast as possible (like opensora eval)",
    )
    parser.add_argument(
        "--no_nirvana",
        action="store_true",
        help="Disable Nirvana cache: every request does full 30-step generation (for A/B time comparison)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="request_throughput_video_teacache_w_nirvana.csv",
        help="log file path",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Max number of worker processes (default: all GPUs). Use 1 for fair Nirvana vs no-Nirvana comparison to avoid GPU contention.",
    )
    args = parser.parse_args()

    os.makedirs(args.video_directory, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices")
    num_workers = args.num_workers if args.num_workers is not None else num_gpus
    num_workers = min(num_workers, num_gpus)

    # Prompts (same as opensora: read_prompt_list uses prompt_en from JSON)
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

    # Request schedule
    seconds_from_start = generate_rapidly_increasing_seconds_from_start(
        len(prompts), min_rate=0.5, max_rate=4
    )
    selected_requests = pd.DataFrame(
        {"prompt": prompts, "seconds_from_start": seconds_from_start}
    )

    # Empty initial cache
    # Bug 4 fix: IndexFlatIP + L2-normalized embeddings → inner product == cosine similarity,
    # so FAISS nearest-neighbour is consistent with the cosine threshold decisions.
    embedding_dim = 768
    index = faiss.IndexFlatIP(embedding_dim)
    final_text_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
    cached_requests = []
    # KMinHeapCache with empty initial state and k_values [5, 10, 15]
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
    # Bug 3 fix: event lets scheduler notify workers to exit cleanly when done.
    done_event = manager.Event()

    clip_model_name = "openai/clip-vit-large-patch14"
    clip_device     = "cuda:0" if torch.cuda.is_available() else "cpu"

    wall_start = time.time()
    scheduler = mp.Process(
        target=request_scheduler_video,
        args=(
            req_queue,
            selected_requests,
            wall_start,
            index,
            cache,
            new_cache_queue,
            cached_requests,
            final_text_embeddings,
            K_VALUES_VIDEO,
            clip_model_name,
            clip_device,
            worker_status,
            done_event,
        ),
        kwargs={
            "log_file": args.log_file,
            "eval_mode": args.eval_mode,
            "no_nirvana": args.no_nirvana,
            "cache_stats": cache_stats,
        },
    )
    scheduler.start()

    workers = []
    for gpu_id in range(num_workers):
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
                args.resolution,
                args.aspect_ratio,
                args.num_frames,
                args.teacache_thresh,
                done_event,
                args.loop,
                args.no_nirvana,
            ),
        )
        p.start()
        workers.append(p)

    total_requests = len(prompts)
    all_latencies = []
    all_processing_times = []
    with tqdm(total=total_requests, desc="Requests", unit="req") as pbar:
        for _ in range(total_requests):
            finish_time, pure_processing_time = latency_queue.get()
            all_latencies.append(finish_time)
            all_processing_times.append(pure_processing_time)
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
