import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from decord import VideoReader, cpu

def extract_frames(video_path, num_frames=8):
    """
    Extracts evenly spaced frames from a video using decord for high performance.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            return None
            
        # Get 'num_frames' evenly spaced indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        
        # decord returns (N, H, W, C) in RGB format, which is perfect for CLIPProcessor
        return frames
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch CLIP Scorer for AI Video Generations")
    parser.add_argument("--video_directory", type=str, default="./video_outputs_teacache", help="Directory containing generated mp4s")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="Where to save the scores")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames to sample per video")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14", help="CLIP model ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Frame batch size for the GPU")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.clip_model} on {device}...")
    
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()

    # Find all mp4s in the directory
    search_pattern = os.path.join(args.video_directory, "*.mp4")
    video_files = glob.glob(search_pattern)
    
    if not video_files:
        print(f"No .mp4 files found in {args.video_directory}")
        return

    print(f"Found {len(video_files)} videos. Starting evaluation...")
    results = []

    with torch.no_grad():
        for video_path in tqdm(video_files, desc="Evaluating Videos"):
            # Extract prompt from filename: "prompt_text-0.mp4"
            filename = os.path.basename(video_path)
            # Reverse split by '-' to separate the loop index '{l}.mp4' from the prompt
            try:
                prompt_str = filename.rsplit('-', 1)[0] 
            except ValueError:
                prompt_str = filename.replace(".mp4", "")

            # 1. Extract Frames
            frames = extract_frames(video_path, num_frames=args.num_frames)
            if frames is None:
                continue
                
            # 2. Process Text and Images
            # We process images in chunks if num_frames is large, but 8 easily fits in one batch
            inputs = processor(
                text=[prompt_str], 
                images=list(frames), # Processor expects a list of arrays or PIL images
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(device)

            # 3. Model Inference
            outputs = model(**inputs)
            
            # 4. Calculate Cosine Similarity (CLIP Score)
            # logits_per_image is shape (num_frames, 1)
            # We scale it back down by dividing by 100 (CLIP logits are multiplied by a logit_scale usually ~100)
            logits_per_image = outputs.logits_per_image
            scores = (logits_per_image / model.logit_scale.exp()).squeeze().tolist()
            
            # Handle edge case where num_frames=1
            if isinstance(scores, float):
                scores = [scores]

            average_score = sum(scores) / len(scores)

            results.append({
                "video_file": filename,
                "prompt": prompt_str,
                "clip_score": round(average_score, 4),
                "frame_scores": [round(s, 4) for s in scores]
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nEvaluation complete! Results saved to {args.output_csv}")
    print(f"Average System CLIP Score: {df['clip_score'].mean():.4f}")

if __name__ == "__main__":
    main()