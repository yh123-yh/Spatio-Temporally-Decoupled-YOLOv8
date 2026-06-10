from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo_label(path: Path) -> torch.Tensor:
    if not path.exists():
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append([float(v) for v in parts[1:5]])
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32).clamp(0, 1)


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def make_temporal_sequence(image: np.ndarray, length: int = 8) -> torch.Tensor:
    frames = []
    h, w = image.shape[:2]
    for _ in range(length):
        angle = random.uniform(-1.5, 1.5)
        scale = random.uniform(0.98, 1.02)
        tx = random.uniform(-0.01, 0.01) * w
        ty = random.uniform(-0.01, 0.01) * h
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        mat[:, 2] += (tx, ty)
        aug = cv2.warpAffine(image, mat, (w, h), borderValue=(114, 114, 114))
        if random.random() < 0.25:
            x1 = random.randint(0, max(w - 16, 1))
            y1 = random.randint(0, max(h - 16, 1))
            x2 = min(w, x1 + random.randint(12, max(w // 8, 16)))
            y2 = min(h, y1 + random.randint(12, max(h // 8, 16)))
            aug[y1:y2, x1:x2] = 80
        frames.append(image_to_tensor(aug))
    return torch.stack(frames, dim=0)


class YoloSequenceDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", imgsz: int = 640, sequence_length: int = 8):
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.sequence_length = sequence_length
        image_dir = self.root / "images" / split
        self.images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not self.images:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path = self.images[idx]
        label_path = self.root / "labels" / self.split / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image = letterbox(image, self.imgsz)
        return {
            "image": image_to_tensor(image),
            "sequence": make_temporal_sequence(image, self.sequence_length),
            "boxes": read_yolo_label(label_path),
            "count": torch.tensor(float(read_yolo_label(label_path).shape[0]), dtype=torch.float32),
            "path": str(image_path),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list[torch.Tensor] | list[str]]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "sequence": torch.stack([b["sequence"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "count": torch.stack([b["count"] for b in batch]),
        "path": [b["path"] for b in batch],
    }
