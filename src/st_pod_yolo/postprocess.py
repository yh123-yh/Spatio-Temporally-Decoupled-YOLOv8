from __future__ import annotations

import torch

from .box_ops import box_iou


def dynamic_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    base_threshold: float = 0.45,
    eta: float = 0.15,
    topk: int = 240,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Density-aware NMS used to retain close pod candidates when scores are stable."""

    if boxes.numel() == 0:
        return boxes, scores
    order = scores.argsort(descending=True)[:topk]
    selected: list[int] = []
    while order.numel() > 0:
        idx = int(order[0])
        selected.append(idx)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[idx : idx + 1], boxes[order[1:]]).squeeze(0)
        threshold = min(0.70, base_threshold + eta * (iou > 0.30).float().mean().item())
        order = order[1:][iou <= threshold]
    if not selected:
        return boxes[:0], scores[:0]
    idx = torch.tensor(selected, device=boxes.device, dtype=torch.long)
    return boxes[idx], scores[idx]


def count_correction(
    detections: list[dict[str, torch.Tensor]],
    duplicate_iou: float = 0.60,
    miss_iou: float = 0.35,
    compensation_weight: float = 0.5,
) -> list[float]:
    """Correct duplicate and short-gap missed counts across an image sequence."""

    corrected: list[float] = []
    previous_boxes: torch.Tensor | None = None
    for det in detections:
        boxes = det["boxes"]
        raw_count = float(boxes.shape[0])
        duplicate_penalty = 0.0
        if boxes.shape[0] > 1:
            iou = box_iou(boxes, boxes)
            duplicate_penalty = float(((iou > duplicate_iou).sum() - boxes.shape[0]).clamp(min=0).item()) * 0.5
        miss_bonus = 0.0
        if previous_boxes is not None and previous_boxes.numel() and boxes.numel():
            best_prev = box_iou(previous_boxes, boxes).max(dim=1).values
            miss_bonus = float((best_prev < miss_iou).sum().item()) * compensation_weight
        corrected.append(max(0.0, raw_count - duplicate_penalty + miss_bonus))
        previous_boxes = boxes.detach()
    return corrected
