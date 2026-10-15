import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
import torchvision
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Building Dataset Cleaner")
    parser.add_argument("--results",required=True, type=Path, help="Directory containing the dataset images")
    parser.add_argument("--ground_truth",required=True,type=Path,help="COCO annotation JSON file")
    return parser.parse_args()
def coco_to_xyxy(box):
    x,y,w,h = box
    return([x,y,x+w,y+h])
def load_coco_annotations(annotation_path: Path) -> tuple[dict[int,dict[str,Any]], dict[int,list[str,Any]], dict[int, str]]:
    with annotation_path.open("r",encoding="utf-8") as file:
        coco: dict[str, Any] = json.load(file)
    images = {
            int(image["id"]): image
            for image in coco.get("images", [])
        }
    annotations_by_image = defaultdict(list)
    for annotation in coco.get("annotations", []):

        image_id = annotation.get("image_id")

        if image_id is not None:
            annotations_by_image[int(image_id)].append(
                annotation
            )
    return images,annotations_by_image

def load_results(results_dir: Path):
    with results_dir.open("r", encoding="utf-8") as file:
        results = json.load(file)
        return results
    
def IOU(gt, pred):
    gt_tensor = torch.tensor([gt], dtype=torch.float32)
    pred_tensor = torch.tensor([pred], dtype=torch.float32)
    iou = torchvision.ops.box_iou(gt_tensor, pred_tensor)
    return(iou)

if __name__ == "__main__":
    args = parse_args()
    results_dir = args.results.expanduser().resolve()
    gt_path = args.ground_truth.expanduser().resolve()
    (coco_images,annotations_by_image) = load_coco_annotations(gt_path)
    coco_by_filename = {str(image["file_name"]): image
                            for image in coco_images.values()}
    results = load_results(results_dir)
    results_by_file_name = {str(result["image"]): result for result in results}
    # Need to map image_id to image name to correlate annotations in gt to results
    tp = 0
    fp = 0
    total_pos = 0
    for image_name, coco_image in coco_by_filename.items():
        result = results_by_file_name[image_name]
        pred_matched = [0] * len(result["detections"])
        image_id = coco_image["id"]
        GT = annotations_by_image[image_id]
        gt_boxes = []
        pred_boxes = []
        for annotation in GT:
            total_pos+=1
            box = coco_to_xyxy(annotation['bbox'])
            gt_boxes.append(box)
        for detection in result["detections"]:
            pred_boxes.append(detection['box'])
        for gt in gt_boxes:
            IOUs = []
            for x in range(len(pred_boxes)):
                IOU_with_pred = IOU(gt, pred_boxes[x])
                IOUs.append({"IOU":IOU_with_pred.tolist()[0][0], "arg":x})
            IOUs = sorted(IOUs, key=lambda x: x["IOU"], reverse=True)
            for iou in IOUs:
                if pred_matched[iou["arg"]] == 0 and iou["IOU"] > 0.5:
                    tp+=1
                    pred_matched[iou["arg"]] = 1
                    break
        for x in range(len(pred_matched)):
            if pred_matched[x] == 0:
                fp+=1
    precision = tp/(tp+fp)
    recall = tp/(total_pos)
    print(f"Precision: {precision}")
    print(f"recall: {recall}")
   