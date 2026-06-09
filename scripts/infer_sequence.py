from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from st_pod_yolo.data import IMAGE_EXTS, image_to_tensor, letterbox, make_temporal_sequence
from st_pod_yolo.models import build_model_from_dict
from st_pod_yolo.postprocess import count_correction, dynamic_nms


def load_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [source]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    ckpt = torch.load(args.weights, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_dict(ckpt.get("config", {}).get("model", {}))
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    detections = []
    for path in load_images(Path(args.source)):
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = letterbox(image, args.imgsz)
        tensor = image_to_tensor(image).unsqueeze(0).to(device)
        sequence = make_temporal_sequence(image, 8).unsqueeze(0).to(device)
        out = model(tensor, sequence)
        probs = out["logits"][0].sigmoid().cpu()
        boxes = out["boxes"][0].cpu()
        mask = probs >= args.conf
        boxes, scores = dynamic_nms(boxes[mask], probs[mask])
        detections.append({"boxes": boxes, "scores": scores})
        print(f"{path}: raw_count={boxes.shape[0]}")
    corrected = count_correction(detections)
    if corrected:
        print("corrected_counts=", [round(x, 2) for x in corrected])


if __name__ == "__main__":
    main()
