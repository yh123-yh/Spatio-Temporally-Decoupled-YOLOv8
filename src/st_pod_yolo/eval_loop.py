from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import YoloSequenceDataset, collate_fn
from .metrics import DetectionStats, count_metrics, match_detections
from .models import build_model_from_dict
from .postprocess import dynamic_nms


@torch.no_grad()
def evaluate(weights: str | Path, data_cfg: dict, split: str = "test", conf: float = 0.25, iou: float = 0.5) -> dict[str, float]:
    ckpt = torch.load(weights, map_location="cpu")
    model_cfg = ckpt.get("config", {}).get("model", {})
    model = build_model_from_dict(model_cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    device = torch.device(data_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    ds = YoloSequenceDataset(data_cfg["root"], split, int(data_cfg.get("imgsz", 1280)), int(data_cfg.get("sequence_length", 8)))
    loader = DataLoader(ds, batch_size=int(data_cfg.get("batch", 1)), shuffle=False, collate_fn=collate_fn)
    pred_counts: list[float] = []
    true_counts: list[float] = []
    stats = DetectionStats()
    for batch in loader:
        out = model(batch["image"].to(device), batch["sequence"].to(device))
        probs = out["logits"].sigmoid().cpu()
        boxes = out["boxes"].cpu()
        for b in range(boxes.shape[0]):
            mask = probs[b] >= conf
            det_boxes, _ = dynamic_nms(boxes[b][mask], probs[b][mask])
            true = batch["boxes"][b]
            pred_counts.append(float(det_boxes.shape[0]))
            true_counts.append(float(true.shape[0]))
            st = match_detections(det_boxes, true, iou)
            stats.tp += st.tp
            stats.fp += st.fp
            stats.fn += st.fn
    result = count_metrics(pred_counts, true_counts)
    result.update({"F1": stats.f1, "Precision": stats.precision, "Recall": stats.recall})
    return result
