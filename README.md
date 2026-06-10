# Spatio-Temporally Decoupled YOLOv8 Pod Counting Reproduction

`Counting Optimization of a Spatio-Temporally Decoupled YOLOv8 Model in Scenes with Dense Pods`

this project provides:

- A YOLOv8-like PyTorch detector with P2 small-object features.
- Spatial structure and temporal association branches.
- Gated pre-head fusion, count regression, dynamic NMS, and count correction.
- Dataset preparation for public/manual YOLO-format pod data plus a synthetic temporal smoke dataset.
- Training, evaluation, inference, ablation, and benchmark scripts.
- A Chinese reproduction guide in `docs/reproduction_zh.md`.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_data.py --dataset synthetic --out data/smoke --num-images 20
python scripts/train.py --config configs/train_smoke.yaml
python scripts/evaluate.py --weights runs/smoke/best.pt --data configs/data/smoke.yaml
python scripts/benchmark.py --weights runs/smoke/best.pt --imgsz 1280
```

To create a larger dense-pod substitute dataset locally:

```powershell
python scripts/prepare_data.py --dataset synthetic --out data/pods_dense --num-images 1000 --imgsz 640 --sequence-length 8 --dense --min-pods 45 --max-pods 160
python scripts/train.py --config configs/train_dense.yaml
```

For the public-data path after Google Drive download:

```powershell
python scripts/prepare_data.py --dataset yolopod --out data/yolopod --download
python scripts/prepare_data.py --dataset manual --raw datasets\yolopod_raw --out data\yolopod_yolo --imgsz 1280 --sequence-length 8
python scripts/train.py --config configs/train_public.yaml
```

If the Google Drive mirror is unavailable, place images and YOLO labels under `data/raw_yolopod`
and run:

```powershell
python scripts/prepare_data.py --dataset manual --raw data/raw_yolopod --out data/yolopod
```

For a downloaded ZIP with LabelImg/Pascal VOC XML annotations:

```powershell
python scripts/prepare_data.py --dataset manual --archive path\to\yolopod.zip --out data/yolopod
```

Source dataset page: https://plantmethods.biomedcentral.com/articles/10.1186/s13007-023-00985-4  
Google Drive folder: https://drive.google.com/drive/folders/1-Ouj8fFG_owOnJtDDGBQ29_gDyCUdu93?usp=sharing

The project also supports the Kaggle UAV soybean pod point-label dataset after conversion:

```powershell
kaggle datasets download -d jiajiali/uav-based-soybean-pod-images -p datasets\kaggle_uav_soybean --unzip
python scripts/prepare_data.py --dataset manual --raw datasets\kaggle_uav_soybean\dataset --out data\kaggle_uav_yolo --imgsz 1280 --sequence-length 8
python scripts/train.py --config configs/train_kaggle_uav.yaml
```

## Main Commands

```powershell
python scripts/prepare_data.py --dataset yolopod --out data/yolopod --download
python scripts/prepare_data.py --dataset manual --raw datasets\yolopod_raw --out data\yolopod_yolo --imgsz 1280 --sequence-length 8
python scripts/train.py --config configs/train_public.yaml
python scripts/evaluate.py --weights runs/public/best.pt --data configs/data/yolopod.yaml
python scripts/infer_sequence.py --weights runs/public/best.pt --source path/to/images_or_video
python scripts/run_ablation.py --config configs/ablation.yaml
python scripts/benchmark.py --weights runs/public/best.pt --imgsz 1280
```

## Notes

The exact reported metrics require the unpublished self-built dataset and the exact
PlantCrop subset used by the authors. 
