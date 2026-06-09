from __future__ import annotations

import argparse
import time

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from st_pod_yolo.models import build_model_from_dict


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        model = build_model_from_dict(ckpt.get("config", {}).get("model", {}))
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model = build_model_from_dict({})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    seq = torch.randn(1, 8, 3, args.imgsz, args.imgsz, device=device)
    for _ in range(args.warmup):
        model(x, seq)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        model(x, seq)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = elapsed / args.iters * 1000
    fps = 1000 / ms
    print({"Params/M": round(count_params(model), 3), "FPS": round(fps, 2), "Single-frame ms": round(ms, 2)})


if __name__ == "__main__":
    main()
