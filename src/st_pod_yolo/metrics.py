from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .box_ops import box_iou


@dataclass
class DetectionStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)


def count_metrics(pred_counts: list[float], true_counts: list[float]) -> dict[str, float]:
    pred = np.asarray(pred_counts, dtype=np.float64)
    true = np.asarray(true_counts, dtype=np.float64)
    err = pred - true
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err**2)))}


def match_detections(pred_boxes: torch.Tensor, true_boxes: torch.Tensor, iou_threshold: float = 0.5) -> DetectionStats:
    stats = DetectionStats()
    if pred_boxes.numel() == 0:
        stats.fn = int(true_boxes.shape[0])
        return stats
    if true_boxes.numel() == 0:
        stats.fp = int(pred_boxes.shape[0])
        return stats
    ious = box_iou(pred_boxes, true_boxes)
    matched_true: set[int] = set()
    for pred_idx in range(pred_boxes.shape[0]):
        best_iou, best_true = ious[pred_idx].max(dim=0)
        j = int(best_true.item())
        if float(best_iou.item()) >= iou_threshold and j not in matched_true:
            stats.tp += 1
            matched_true.add(j)
        else:
            stats.fp += 1
    stats.fn += true_boxes.shape[0] - len(matched_true)
    return stats
