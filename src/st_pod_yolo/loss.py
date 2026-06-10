from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .box_ops import box_iou


class PodCountingLoss(nn.Module):
    def __init__(self, box_weight: float = 5.0, obj_weight: float = 1.0, count_weight: float = 0.2):
        super().__init__()
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.count_weight = count_weight

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[torch.Tensor], counts: torch.Tensor) -> dict[str, torch.Tensor]:
        boxes = outputs["boxes"]
        logits = outputs["logits"]
        pred_count = outputs["count"]
        obj_target = torch.zeros_like(logits)
        box_loss = boxes.new_tensor(0.0)
        matched = 0
        with torch.no_grad():
            for b, true in enumerate(targets):
                if true.numel() == 0:
                    continue
                iou = box_iou(boxes[b], true.to(boxes.device))
                best_pred = iou.argmax(dim=0)
                obj_target[b, best_pred] = 1.0
        for b, true in enumerate(targets):
            true = true.to(boxes.device)
            if true.numel() == 0:
                continue
            iou = box_iou(boxes[b], true)
            best_pred = iou.argmax(dim=0)
            box_loss = box_loss + F.l1_loss(boxes[b, best_pred], true, reduction="sum")
            matched += true.shape[0]
        if matched:
            box_loss = box_loss / matched
        obj_loss = F.binary_cross_entropy_with_logits(logits, obj_target)
        count_loss = F.mse_loss(pred_count, counts.to(pred_count.device))
        total = self.box_weight * box_loss + self.obj_weight * obj_loss + self.count_weight * count_loss
        return {"loss": total, "box_loss": box_loss.detach(), "obj_loss": obj_loss.detach(), "count_loss": count_loss.detach()}
