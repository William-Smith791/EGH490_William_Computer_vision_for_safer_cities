#!/usr/bin/env python3
"""Test Grounding DINO building detection on all images in a directory.

The defaults follow NHRA Final Report, Building Detection (Appendix 1).
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-base"
PROMPT = "building. house. apartment building. facade."
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def overlap(a, b):
    """Return IoU and smaller-box containment (intersection / smaller area)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return (inter / union if union else 0.0,
            inter / min(area_a, area_b) if min(area_a, area_b) else 0.0)


def postprocess(predictions, image_size, args):
    """Confidence filter, top raw proposals, greedy duplicate suppression, top 3, area filter."""
    width, height = image_size
    candidates = sorted(
        (p for p in predictions if p["score"] >= args.score_threshold),
        key=lambda p: p["score"], reverse=True,
    )[:args.max_raw]

    kept = []
    for candidate in candidates:
        if all(not (overlap(candidate["box"], other["box"])[0] > args.iou_threshold
                    or overlap(candidate["box"], other["box"])[1] > args.containment_threshold)
               for other in kept):
            kept.append(candidate)

    kept = kept[:args.max_detections]
    return [p for p in kept if
            max(0, p["box"][2] - p["box"][0]) *
            max(0, p["box"][3] - p["box"][1]) >= args.min_area * width * height]


def visualise(image, detections, filename):
    # Import only when needed: no plotting dependency at inference time.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(image)
    ax.set_title(f"{filename} — {len(detections)} building(s) | close window for next image")
    ax.axis("off")
    for p in detections:
        x1, y1, x2, y2 = p["box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor="lime", linewidth=2.5))
        ax.text(x1, max(0, y1 - 5), f'{p["score"]:.2f} ({p["label"]})',
                color="black", fontsize=9,
                bbox={"facecolor": "lime", "alpha": 0.85, "pad": 2})
    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-folder", required=True, type=Path, help="Folder containing images")
    parser.add_argument("--visualise", action="store_true", help="Show detections in a window; close to continue")
    parser.add_argument("--output-json", type=Path, help="Optional path for detection results in JSON")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.35)
    parser.add_argument("--containment-threshold", type=float, default=0.75)
    parser.add_argument("--max-raw", type=int, default=30)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--min-area", type=float, default=0.04,
                        help="Minimum detection area as a fraction of the image")
    args = parser.parse_args()

    image_folder = args.input_folder.expanduser()
    if not image_folder.is_dir():
        parser.error(f"Not a directory: {image_folder}")
    image_paths = sorted(p for p in image_folder.iterdir()
                         if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not image_paths:
        parser.error(f"No supported images found in {image_folder}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    results = []
    for index, path in enumerate(image_paths, start=1):
        try:
            with Image.open(path) as loaded:
                image = ImageOps.exif_transpose(loaded).convert("RGB")
            inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model(**inputs)
            raw = processor.post_process_grounded_object_detection(
                output, inputs.input_ids, box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            predictions = [
                {"box": [round(float(x), 2) for x in box.tolist()],
                 "score": round(float(score), 5), "label": str(label)}
                for box, score, label in zip(raw["boxes"], raw["scores"], raw["labels"])
            ]
            detections = postprocess(predictions, image.size, args)
            results.append({"image": path.name, "width": image.width,
                            "height": image.height, "detections": detections})
            print(f"[{index}/{len(image_paths)}] {path.name}: {len(detections)} building(s)")
            if args.visualise:
                visualise(image, detections, path.name)
        except Exception as exc:
            print(f"[{index}/{len(image_paths)}] ERROR {path.name}: {exc}")
            results.append({"image": path.name, "error": str(exc)})

    if args.output_json:
        output_path = args.output_json.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved: {output_path}")
    print(f"Finished: {sum('detections' in x for x in results)}/{len(image_paths)} images processed successfully")


if __name__ == "__main__":
    main()
