from __future__ import annotations

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import YoloSequenceDataset, collate_fn
from .loss import PodCountingLoss
from .models import build_model_from_dict


def train_from_config(cfg: dict) -> Path:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(cfg.get("out_dir", "runs/public"))
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = cfg["data"]["root"]
    imgsz = int(cfg.get("imgsz", 1280))
    train_ds = YoloSequenceDataset(data_root, "train", imgsz, int(cfg.get("sequence_length", 8)))
    val_ds = YoloSequenceDataset(data_root, "val", imgsz, int(cfg.get("sequence_length", 8)))
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch", 2)), shuffle=True, num_workers=int(cfg.get("workers", 0)), collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch", 2)), shuffle=False, num_workers=int(cfg.get("workers", 0)), collate_fn=collate_fn)
    model = build_model_from_dict(cfg.get("model", {})).to(device)
    criterion = PodCountingLoss(**cfg.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 0.001)), weight_decay=float(cfg.get("weight_decay", 0.0005)))
    best = float("inf")
    epochs = int(cfg.get("epochs", 120))
    grad_accum = int(cfg.get("grad_accum", 1))
    history_path = out_dir / "metrics_history.jsonl"
    for epoch in range(epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            image = batch["image"].to(device)
            sequence = batch["sequence"].to(device)
            counts = batch["count"].to(device)
            outputs = model(image, sequence)
            loss_dict = criterion(outputs, batch["boxes"], counts)
            (loss_dict["loss"] / grad_accum).backward()
            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss_dict["loss"].detach().cpu())
            pbar.set_postfix(loss=running / step)
        val_mae = _quick_val_mae(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": running / max(len(train_loader), 1), "val_count_mae": val_mae}
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(record)
        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch + 1, "val_mae": val_mae}
        torch.save(ckpt, out_dir / "last.pt")
        if val_mae <= best:
            best = val_mae
            torch.save(ckpt, out_dir / "best.pt")
    return out_dir / "best.pt"


@torch.no_grad()
def _quick_val_mae(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errors = []
    for batch in loader:
        outputs = model(batch["image"].to(device), batch["sequence"].to(device))
        errors.extend((outputs["count"].cpu() - batch["count"]).abs().tolist())
    return float(sum(errors) / max(len(errors), 1))
