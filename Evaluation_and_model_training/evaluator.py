import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Building Dataset Cleaner")
    parser.add_argument("--results",required=True, type=Path, help="Directory containing the dataset images")
    parser.add_argument("--ground_truth",required=True,type=Path,help="COCO annotation JSON file")
    return parser.parse_args()

def load_coco_annotations(annotation_path: Path) -> tuple[dict[int,dict[str,Any]], dict[int,list[str,Any]], dict[int, str]]:
    with annotation_path.open("r",encoding="utf-8") as file:
        coco: dict[str, Any] = json.load(file)
    annotations_by_image = defaultdict(list)
    for annotation in coco.get("annotations", []):

        image_id = annotation.get("image_id")

        if image_id is not None:
            annotations_by_image[int(image_id)].append(
                annotation
            )
    return annotations_by_image

def load_results(results_dir: Path):
    with results_dir.open("r", encoding="utf-8") as file:
        results = json.load(file)
        return results

if __name__ == "__main__":
    args = parse_args()
    results_dir = args.results.expanduser().resolve()
    gt_path = args.ground_truth.expanduser().resolve()
    annotations_by_image = load_coco_annotations(gt_path)
    results = load_results(results_dir)
    # coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()