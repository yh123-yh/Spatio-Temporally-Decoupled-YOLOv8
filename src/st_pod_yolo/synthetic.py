from __future__ import annotations

import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np


def create_synthetic_dataset(
    out: str | Path,
    num_images: int = 20,
    imgsz: int = 640,
    seed: int = 42,
    min_pods: int = 6,
    max_pods: int = 22,
    dense: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    out = Path(out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = {"train": int(num_images * 0.7), "val": max(1, int(num_images * 0.2))}
    counts["test"] = max(1, num_images - counts["train"] - counts["val"])
    idx = 0
    for split, n in counts.items():
        for _ in range(n):
            image = np.full((imgsz, imgsz, 3), (42, 72, 44), dtype=np.uint8)
            labels: list[str] = []
            pod_count = random.randint(min_pods, max_pods)
            centers = _sample_dense_centers(pod_count) if dense else [(random.uniform(0.08, 0.92), random.uniform(0.08, 0.92)) for _ in range(pod_count)]
            for cx, cy in centers:
                length = random.uniform(0.025, 0.075) if dense else random.uniform(0.035, 0.085)
                width = random.uniform(0.008, 0.020) if dense else random.uniform(0.010, 0.026)
                angle = random.uniform(0, 180)
                color = (
                    random.randint(70, 155),
                    random.randint(95, 185),
                    random.randint(45, 105),
                )
                center = (int(cx * imgsz), int(cy * imgsz))
                axes = (max(3, int(length * imgsz / 2)), max(2, int(width * imgsz / 2)))
                cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)
                labels.append(f"0 {cx:.6f} {cy:.6f} {length:.6f} {width:.6f}")
            occluder_count = random.randint(12, 28) if dense else random.randint(5, 12)
            for _ in range(occluder_count):
                p1 = (random.randint(0, imgsz), random.randint(0, imgsz))
                p2 = (random.randint(0, imgsz), random.randint(0, imgsz))
                cv2.line(image, p1, p2, (25, random.randint(70, 115), 35), random.randint(2, 9 if dense else 6))
            if dense:
                for _ in range(random.randint(4, 10)):
                    x1 = random.randint(0, imgsz - 1)
                    y1 = random.randint(0, imgsz - 1)
                    x2 = min(imgsz, x1 + random.randint(imgsz // 20, imgsz // 7))
                    y2 = min(imgsz, y1 + random.randint(imgsz // 20, imgsz // 7))
                    alpha = random.uniform(0.25, 0.45)
                    patch = image[y1:y2, x1:x2]
                    leaf = np.full_like(patch, (25, random.randint(75, 125), 35))
                    image[y1:y2, x1:x2] = cv2.addWeighted(patch, 1 - alpha, leaf, alpha, 0)
            name = f"synthetic_{idx:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), image)
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(labels), encoding="utf-8")
            idx += 1


def _sample_dense_centers(n: int) -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    cluster_count = random.randint(3, 8)
    cluster_roots = [(random.uniform(0.12, 0.88), random.uniform(0.12, 0.88)) for _ in range(cluster_count)]
    for _ in range(n):
        rx, ry = random.choice(cluster_roots)
        cx = min(0.96, max(0.04, random.gauss(rx, 0.055)))
        cy = min(0.96, max(0.04, random.gauss(ry, 0.055)))
        centers.append((cx, cy))
    return centers


def split_manual_yolo(raw: str | Path, out: str | Path, seed: int = 42) -> None:
    raw = Path(raw)
    out = Path(out)
    image_candidates = sorted(
        p for p in raw.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_candidates:
        raise FileNotFoundError(f"No images found under {raw}")
    random.Random(seed).shuffle(image_candidates)
    n = len(image_candidates)
    splits = {
        "train": image_candidates[: int(n * 0.7)],
        "val": image_candidates[int(n * 0.7) : int(n * 0.9)],
        "test": image_candidates[int(n * 0.9) :],
    }
    for split, images in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in images:
            shutil.copy2(image, out / "images" / split / image.name)
            yolo_text = _read_or_convert_label(raw, image)
            dst = out / "labels" / split / f"{image.stem}.txt"
            dst.write_text(yolo_text, encoding="utf-8")


def extract_archive(archive: str | Path, out: str | Path) -> Path:
    archive = Path(archive)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    out_resolved = out.resolve()
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (out / member).resolve()
                if not str(target).startswith(str(out_resolved)):
                    raise ValueError(f"Unsafe archive member path: {member}")
            zf.extractall(out)
    elif archive.suffix.lower() in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = (out / member.name).resolve()
                if not str(target).startswith(str(out_resolved)):
                    raise ValueError(f"Unsafe archive member path: {member.name}")
            tf.extractall(out)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")
    return out


def _read_or_convert_label(raw: Path, image: Path) -> str:
    txt_candidates = [
        image.with_suffix(".txt"),
        raw / "labels" / f"{image.stem}.txt",
        raw / "Labels" / f"{image.stem}.txt",
    ]
    for label in txt_candidates:
        if label.exists():
            return label.read_text(encoding="utf-8").strip()
    xml_candidates = [
        image.with_suffix(".xml"),
        raw / "annotations" / f"{image.stem}.xml",
        raw / "Annotations" / f"{image.stem}.xml",
        raw / "xml" / f"{image.stem}.xml",
    ]
    for xml in xml_candidates:
        if xml.exists():
            return _voc_xml_to_yolo(xml)
    json_candidates = [
        image.with_suffix(".json"),
        raw / "annotations" / f"{image.stem}.json",
        raw / "Annotations" / f"{image.stem}.json",
        raw / "json" / f"{image.stem}.json",
    ]
    for json_path in json_candidates:
        if json_path.exists():
            return _labelme_json_to_yolo(json_path)
    return ""


def _voc_xml_to_yolo(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"VOC XML missing size: {xml_path}")
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid VOC image size in {xml_path}")
    lines = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        cx = ((xmin + xmax) / 2) / width
        cy = ((ymin + ymax) / 2) / height
        bw = max(0.0, xmax - xmin) / width
        bh = max(0.0, ymax - ymin) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def _labelme_json_to_yolo(json_path: Path, point_box_px: float = 20.0) -> str:
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = float(data.get("imageWidth", 0))
    height = float(data.get("imageHeight", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid LabelMe image size in {json_path}")
    lines = []
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        if not points:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        if shape.get("shape_type") == "point" or (len(xs) == 1 and len(ys) == 1):
            half = point_box_px / 2.0
            xmin, xmax = xs[0] - half, xs[0] + half
            ymin, ymax = ys[0] - half, ys[0] + half
        else:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        xmin, ymin = max(0.0, xmin), max(0.0, ymin)
        xmax, ymax = min(width, xmax), min(height, ymax)
        bw = max(1.0, xmax - xmin)
        bh = max(1.0, ymax - ymin)
        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}")
    return "\n".join(lines)
