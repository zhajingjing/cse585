import os
import glob
import torch
import lpips
import numpy as np
import pandas as pd
from tqdm import tqdm
from decord import VideoReader, cpu
from itertools import combinations

def extract_frames_tensor(video_path, num_frames=8):
    """
    Extracts frames using decord and formats them for LPIPS.
    LPIPS expects PyTorch tensors in the shape (N, C, H, W) normalized to [-1, 1].
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return None
            
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames_np = vr.get_batch(indices).asnumpy() # Shape: (N, H, W, C), RGB, [0, 255]
        
        # Convert to tensor: (N, C, H, W)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        
        # Normalize from [0, 255] to [-1.0, 1.0] (Required by LPIPS)
        frames_tensor = (frames_tensor / 127.5) - 1.0
        
        return frames_tensor
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None

def calculate_lpips_between_videos(loss_fn, frames_a, frames_b, device):
    """Computes the average LPIPS score across matching frames of two videos."""
    frames_a = frames_a.to(device)
    frames_b = frames_b.to(device)
    
    # lpips processes batches of images. Shape: (num_frames, 3, H, W)
    with torch.no_grad():
        # returns tensor of shape (num_frames, 1, 1, 1)
        distances = loss_fn(frames_a, frames_b)
        
    return distances.mean().item()

def evaluate_diversity(directory, loss_fn, device, num_frames):
    """
    Calculates Pairwise LPIPS between all videos in a single directory.
    Lower score = Videos are identical (Bad for diversity).
    Higher score = Videos are visually unique.
    """
    video_files = glob.glob(os.path.join(directory, "*.mp4"))
    if len(video_files) < 2:
        print("Need at least 2 videos to calculate pairwise diversity.")
        return

    print(f"Loading {len(video_files)} videos into memory for pairwise comparison...")
    video_tensors = {}
    for vf in tqdm(video_files, desc="Extracting"):
        tensor = extract_frames_tensor(vf, num_frames)
        if tensor is not None:
            video_tensors[vf] = tensor

    valid_files = list(video_tensors.keys())
    pairs = list(combinations(valid_files, 2))
    
    results = []
    print(f"Calculating LPIPS for {len(pairs)} pairs...")
    
    for vf1, vf2 in tqdm(pairs, desc="Scoring Pairs"):
        score = calculate_lpips_between_videos(loss_fn, video_tensors[vf1], video_tensors[vf2], device)
        results.append({
            "video_1": os.path.basename(vf1),
            "video_2": os.path.basename(vf2),
            "lpips_distance": round(score, 4)
        })

    df = pd.DataFrame(results)
    print("\n--- DIVERSITY RESULTS (Pairwise LPIPS) ---")
    print("Higher is better (more diverse).")
    print(f"Average Pairwise Diversity: {df['lpips_distance'].mean():.4f}")
    df.to_csv("lpips_diversity_results.csv", index=False)
    print("Saved to lpips_diversity_results.csv")

def evaluate_quality(ref_dir, cache_dir, loss_fn, device, num_frames):
    """
    Calculates LPIPS between a Cold Generation (Reference) and a Cached Generation.
    Lower score = Cache looks identical to the perfect cold run (Good for quality).
    """
    ref_files = glob.glob(os.path.join(ref_dir, "*.mp4"))
    
    results = []
    print(f"Comparing {len(ref_files)} Reference vs. Cached videos...")
    
    for ref_path in tqdm(ref_files, desc="Comparing"):
        filename = os.path.basename(ref_path)
        cache_path = os.path.join(cache_dir, filename)
        
        if not os.path.exists(cache_path):
            continue
            
        ref_tensor = extract_frames_tensor(ref_path, num_frames)
        cache_tensor = extract_frames_tensor(cache_path, num_frames)
        
        if ref_tensor is not None and cache_tensor is not None:
            score = calculate_lpips_between_videos(loss_fn, ref_tensor, cache_tensor, device)
            results.append({
                "video": filename,
                "lpips_degradation": round(score, 4)
            })

    if not results:
        print("No matching file pairs found between the two directories.")
        return

    df = pd.DataFrame(results)
    print("\n--- DEGRADATION RESULTS (Reference vs. Cache LPIPS) ---")
    print("Lower is better (cache did not destroy quality).")
    print(f"Average Degradation: {df['lpips_degradation'].mean():.4f}")
    df.to_csv("lpips_degradation_results.csv", index=False)
    print("Saved to lpips_degradation_results.csv")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LPIPS Evaluator for Cache Diversity and Degradation")
    parser.add_argument("--mode", type=str, choices=["diversity", "quality"], required=True, 
                        help="'diversity' compares a folder to itself. 'quality' compares a cold folder to a cached folder.")
    parser.add_argument("--dir1", type=str, required=True, help="Target directory (or Reference directory for 'quality' mode)")
    parser.add_argument("--dir2", type=str, default="", help="Cached directory (Only needed for 'quality' mode)")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames to sample per video")
    parser.add_argument("--net", type=str, default="vgg", choices=["vgg", "alex"], help="LPIPS backbone (vgg is standard for generation)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LPIPS ({args.net} backbone) on {device}...")
    
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net=args.net).to(device)
    loss_fn.eval()

    if args.mode == "diversity":
        evaluate_diversity(args.dir1, loss_fn, device, args.num_frames)
    elif args.mode == "quality":
        if not args.dir2:
            print("Error: You must provide --dir2 for quality mode.")
            return
        evaluate_quality(args.dir1, args.dir2, loss_fn, device, args.num_frames)

if __name__ == "__main__":
    main()
