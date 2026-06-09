from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from st_pod_yolo.config import save_yaml
from st_pod_yolo.synthetic import create_synthetic_dataset, extract_archive, split_manual_yolo


YOLOPOD_ARTICLE = "https://plantmethods.biomedcentral.com/articles/10.1186/s13007-023-00985-4"
YOLOPOD_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1-Ouj8fFG_owOnJtDDGBQ29_gDyCUdu93?usp=sharing"


def _write_data_yaml(out: Path, imgsz: int, sequence_length: int) -> None:
    save_yaml(
        {
            "root": str(out.as_posix()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": ["pod"],
            "imgsz": imgsz,
            "sequence_length": sequence_length,
            "batch": 1,
        },
        ROOT / "configs" / "data" / f"{out.name}.yaml",
    )


def prepare_yolopod(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if args.download:
        out.mkdir(parents=True, exist_ok=True)
        url = args.google_drive_url or YOLOPOD_DRIVE_FOLDER
        cmd = [sys.executable, "-m", "gdown", "--folder", "--continue", url, "-O", str(out)]
        if args.no_check_certificate:
            cmd.insert(4, "--no-check-certificate")
        subprocess.check_call(cmd)
        print("Downloaded Google Drive folder to", out)
        print("If needed, arrange the downloaded files as YOLO images+labels, then rerun with --dataset manual --raw <folder>.")
        return
    note = f"""YOLO POD public-data preparation needs the dataset package from:
{YOLOPOD_ARTICLE}
Google Drive:
{YOLOPOD_DRIVE_FOLDER}

The article describes the public soybean pod counting data, but there is no stable direct file ID
for every nested file encoded in this repository. Use one of these routes:

1. Download the dataset from the Google Drive folder above, or run:
   python scripts/prepare_data.py --dataset yolopod --out {out} --download
2. Arrange it as:
   raw/
     images/*.jpg
     labels/*.txt   # YOLO: class cx cy w h, normalized
3. Run:
   python scripts/prepare_data.py --dataset manual --raw raw --out {out}

For a runnable smoke test now:
   python scripts/prepare_data.py --dataset synthetic --out data/smoke --num-images 20
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "DATASET_DOWNLOAD_NOTE.txt").write_text(note, encoding="utf-8")
    _write_data_yaml(out, args.imgsz, args.sequence_length)
    print(note)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["yolopod", "manual", "synthetic"], default="synthetic")
    parser.add_argument("--out", default="data/smoke")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--min-pods", type=int, default=6)
    parser.add_argument("--max-pods", type=int, default=22)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--google-drive-url", default=None)
    parser.add_argument("--no-check-certificate", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if args.dataset == "synthetic":
        create_synthetic_dataset(out, args.num_images, args.imgsz, min_pods=args.min_pods, max_pods=args.max_pods, dense=args.dense)
        _write_data_yaml(out, args.imgsz, args.sequence_length)
        print(f"Synthetic dataset written to {out}")
    elif args.dataset == "manual":
        raw = args.raw
        if args.archive:
            raw = str(extract_archive(args.archive, out.parent / "raw_extracted"))
        if not raw:
            raise SystemExit("--raw or --archive is required for --dataset manual")
        split_manual_yolo(raw, out)
        _write_data_yaml(out, args.imgsz, args.sequence_length)
        print(f"Manual YOLO dataset split written to {out}")
    else:
        prepare_yolopod(args)


if __name__ == "__main__":
    main()
