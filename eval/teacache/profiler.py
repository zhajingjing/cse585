import time
import torch
import json
import numpy as np

class ServingProfiler:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_steps_skipped": 0,
            "latencies_sec": [],
            "cache_search_overhead_sec": [],
            "peak_vram_mb": 0
        }

    def start_timer(self):
        return time.perf_counter()

    def record_request(self, latency, cache_hit, steps_skipped, search_overhead):
        self.metrics["total_requests"] += 1
        self.metrics["latencies_sec"].append(latency)
        self.metrics["cache_search_overhead_sec"].append(search_overhead)
        
        if cache_hit:
            self.metrics["cache_hits"] += 1
            self.metrics["total_steps_skipped"] += steps_skipped

        # Track peak VRAM usage
        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.metrics["peak_vram_mb"] = max(self.metrics["peak_vram_mb"], current_peak)

    def print_summary(self):
        reqs = self.metrics["total_requests"]
        if reqs == 0:
            print("No requests processed.")
            return

        hit_rate = (self.metrics["cache_hits"] / reqs) * 100
        avg_latency = np.mean(self.metrics["latencies_sec"])
        avg_overhead = np.mean(self.metrics["cache_search_overhead_sec"])
        
        print("\n=== SERVING SYSTEM PERFORMANCE ===")
        print(f"Total Requests Processed: {reqs}")
        print(f"Average End-to-End Latency: {avg_latency:.2f} seconds/video")
        print(f"Peak VRAM Usage: {self.metrics['peak_vram_mb']:.0f} MB")
        
        print("\n=== CACHE PERFORMANCE ===")
        print(f"Cache Hit Rate: {hit_rate:.1f}% ({self.metrics['cache_hits']}/{reqs})")
        print(f"Total Denoising Steps Skipped: {self.metrics['total_steps_skipped']}")
        print(f"Avg Cache Search Overhead: {avg_overhead * 1000:.2f} ms/request")
        
        # The critical ROI calculation:
        if avg_latency > 0:
            throughput = 60 / avg_latency
            print(f"Estimated Throughput: {throughput:.2f} videos / minute")

    def save_to_disk(self, filepath="serving_metrics.json"):
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=4)
