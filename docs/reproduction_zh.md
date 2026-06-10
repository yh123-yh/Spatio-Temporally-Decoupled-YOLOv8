# 稠密豆荚场景时空解耦 YOLOv8 计数模型复现说明

## 1. 复现边界

本工程复现论文《Counting Optimization of a Spatio-Temporally Decoupled YOLOv8 Model in Scenes with Dense Pods》的核心方法链路：

- P2 细粒度小目标分支；
- 空间结构分支，增强边界、邻域和遮挡关系；
- 时序关联分支，编码 8/16 帧内的中心漂移、尺度波动和短时可见性；
- 检测头前门控融合；
- 候选框动态 NMS；
- 重复计数和漏计数修正；
- 定位损失、目标置信度损失和数量回归损失的联合优化。

论文没有提供自建数据集和源代码，PlantCrop 子集也没有可核验的公开入口。因此本工程默认使用公开 YOLO POD 大豆豆荚数据或合成时序数据完成方法级复现。论文表 3-6 的数值只作为论文报告值，不能在缺少原始数据时保证完全一致。

## 2. 环境安装

```powershell
python -m pip install -r requirements.txt
```

当前工程可在 Windows + NVIDIA GPU 上运行。若显存不足，先使用 `configs/train_smoke.yaml` 或将 `configs/train_public.yaml` 中的 `batch` 调小。

## 3. 数据准备

### 3.1 合成冒烟数据

用于检查工程是否完整可跑：

```powershell
python scripts/prepare_data.py --dataset synthetic --out data/smoke --num-images 20 --imgsz 320 --sequence-length 4
```

该命令会生成 YOLO 格式数据，并写出 `configs/data/smoke.yaml`。

### 3.2 本地稠密豆荚替代数据

当外部数据源无法下载时，可以直接生成一套稠密遮挡数据：

```powershell
python scripts/prepare_data.py --dataset synthetic --out data/pods_dense --num-images 1000 --imgsz 640 --sequence-length 8 --dense --min-pods 45 --max-pods 160
```

该数据集包含聚簇分布、重叠豆荚、枝叶遮挡和 YOLO 框标注，可用于跑完整训练和消融流程。它不是论文真实数据，只用于让复现工程在无外部数据时完整闭环。

### 3.3 公开 YOLO POD 数据

论文替代数据优先选择 YOLO POD：  
https://plantmethods.biomedcentral.com/articles/10.1186/s13007-023-00985-4

论文数据可用性声明给出的 Google Drive：  
https://drive.google.com/drive/folders/1-Ouj8fFG_owOnJtDDGBQ29_gDyCUdu93?usp=sharing

可先尝试自动下载：

```powershell
python scripts/prepare_data.py --dataset yolopod --out data/yolopod --download
```

本机已下载到：

```text
data/yolopod/chongzhou2021_polygon.zip
data/yolopod/pod_annotation.zip
data/yolopod/pod_images.zip
```

本机已解压并转换为：

```text
data/yolopod_yolo
```

统计结果：

| split | images | boxes | avg boxes/image |
| --- | ---: | ---: | ---: |
| train | 1013 | 50701 | 50.05 |
| val | 290 | 14591 | 50.31 |
| test | 145 | 7417 | 51.15 |

如果已经下载并整理为 YOLO 格式：

```text
raw_yolopod/
  images/*.jpg
  labels/*.txt
```

执行：

```powershell
python scripts/prepare_data.py --dataset manual --raw raw_yolopod --out data/yolopod --imgsz 1280 --sequence-length 8
```

如果下载到的是压缩包，或标注是 LabelImg/Pascal VOC XML，也可以直接执行：

```powershell
python scripts/prepare_data.py --dataset manual --archive path\to\yolopod.zip --out data/yolopod --imgsz 1280 --sequence-length 8
```

脚本会自动解压、查找图片、把 XML 框标注转换为 YOLO txt，并按 7:2:1 划分。

### 3.4 Kaggle UAV 大豆豆荚数据

如果使用 Kaggle 的 `uav-based-soybean-pod-images`，数据是 BMP 图片 + LabelMe JSON 点标注。工程会把每个点扩展为一个小 YOLO 框：

```powershell
kaggle datasets download -d jiajiali/uav-based-soybean-pod-images -p datasets\kaggle_uav_soybean --unzip
python scripts/prepare_data.py --dataset manual --raw datasets\kaggle_uav_soybean\dataset --out data\kaggle_uav_yolo --imgsz 1280 --sequence-length 8
python scripts\train.py --config configs\train_kaggle_uav.yaml
```

该数据更偏“点标注计数”，检测框是由点坐标构造出的近似框，适合用于计数流程验证。

如果暂时没有数据，可执行：

```powershell
python scripts/prepare_data.py --dataset yolopod --out data/yolopod
```

工程会生成下载说明和数据配置占位文件。

## 4. 训练

合成数据快速训练：

```powershell
python scripts/train.py --config configs/train_smoke.yaml
```

公开数据完整训练：

```powershell
python scripts/train.py --config configs/train_public.yaml
```

默认训练设置贴近论文：

- 输入尺寸：1280 x 1280；
- epoch：120；
- optimizer：AdamW；
- learning rate：0.001；
- weight decay：0.0005；
- 时序窗口：8 帧；
- 默认 batch：2，梯度累积 4 步，等效 batch 8。

## 5. 评估与推理

评估：

```powershell
python scripts/evaluate.py --weights runs/public/best.pt --data configs/data/yolopod.yaml
```

输出字段包括：

- `MAE`：计数平均绝对误差；
- `RMSE`：计数均方根误差；
- `F1`、`Precision`、`Recall`：检测匹配指标。

序列推理：

```powershell
python scripts/infer_sequence.py --weights runs/public/best.pt --source path/to/images_or_video
```

复杂度和速度：

```powershell
python scripts/benchmark.py --weights runs/public/best.pt --imgsz 1280
```

## 6. 消融实验

```powershell
python scripts/run_ablation.py --config configs/ablation.yaml
```

消融顺序与论文描述一致：

1. YOLOv8n baseline；
2. + P2 branch；
3. + multi-scale fusion；
4. + dynamic NMS；
5. + count correction；
6. + temporal branch；
7. + full joint loss。

结果会写入 `runs/ablation/results.json`。

## 7. 工程实现说明

`src/st_pod_yolo/models.py` 中的 `STPodYOLO` 是主模型。它包含 YOLOv8 风格主干、P2 小目标保真分支、空间结构分支、时序关联分支、门控融合模块、检测头和数量回归头。

`src/st_pod_yolo/postprocess.py` 实现动态 NMS 和跨帧计数修正。动态 NMS 根据局部重叠密度放宽或收紧抑制阈值；计数修正会对高 IoU 重复候选做惩罚，并对相邻帧短暂消失的候选做补偿。

`src/st_pod_yolo/loss.py` 实现联合损失。由于论文 DOCX 中公式对象不可完整提取，工程采用可训练的等价形式：框回归 L1、目标 BCE 和数量 MSE。

## 8. 与论文指标的关系

论文报告的 Proposed model 指标为：

| 数据集 | MAE | RMSE | F1 |
| --- | ---: | ---: | ---: |
| Self-built | 4.2 | 6.8 | 0.91 |
| PlantCrop subset | 3.1 | 5.0 | 0.94 |
| Synthetic sequence | 6.4 | 8.9 | 0.88 |

本工程在公开替代数据上的结果会不同。若要严格复现上述指标，需要获得论文作者的自建数据集、PlantCrop 子集划分、完整标注和原始训练日志。

## 9. 常见问题

### `ultralytics` 未安装是否能跑？

可以。本工程核心模型使用原生 PyTorch 实现，`ultralytics` 主要作为环境兼容和后续扩展依赖。

### 为什么不是直接修改官方 YOLOv8 源码？

当前工作区没有官方源码，且论文没有发布实现。为了保证复现工程可移植，本项目把论文模块实现为独立 PyTorch 包，同时保留 YOLOv8 风格的多尺度检测接口。

### 显存不够怎么办？

把 `imgsz` 降到 640 或 320，把 `batch` 设为 1，并保留 `grad_accum`。
