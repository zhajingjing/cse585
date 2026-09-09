# Common video metrics

This directory contains CLIP, LPIPS, PSNR, and SSIM evaluation utilities.

The standalone video evaluators can be run from the project root:

```bash
python eval/teacache/common_metrics/evaluate_clip.py --help
python eval/teacache/common_metrics/evaluate_lpips.py --help
```

The original PSNR, SSIM, and LPIPS metric scripts were adapted from
[common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality).
