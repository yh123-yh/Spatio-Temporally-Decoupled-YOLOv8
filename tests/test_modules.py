from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st_pod_yolo.models import STPodYOLO, ModelConfig
from st_pod_yolo.postprocess import count_correction, dynamic_nms


def test_model_shapes() -> None:
    model = STPodYOLO(ModelConfig(width=0.25))
    x = torch.randn(2, 3, 128, 128)
    seq = torch.randn(2, 4, 3, 128, 128)
    out = model(x, seq)
    assert out["boxes"].shape[0] == 2
    assert out["boxes"].shape[-1] == 4
    assert out["logits"].shape[:2] == out["boxes"].shape[:2]
    assert out["count"].shape == (2,)


def test_dynamic_nms_keeps_best() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.51, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]])
    scores = torch.tensor([0.9, 0.7, 0.8])
    kept_boxes, kept_scores = dynamic_nms(boxes, scores, base_threshold=0.45)
    assert kept_boxes.shape[0] >= 2
    assert torch.isclose(kept_scores.max(), torch.tensor(0.9))


def test_count_correction_sequence() -> None:
    detections = [
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.51, 0.5, 0.2, 0.2]])},
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
    ]
    corrected = count_correction(detections)
    assert len(corrected) == 2
    assert corrected[0] <= 2


if __name__ == "__main__":
    test_model_shapes()
    test_dynamic_nms_keeps_best()
    test_count_correction_sequence()
    print("module tests passed")
