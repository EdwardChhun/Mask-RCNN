# Mask R-CNN — TACO Litter Detection

Instance segmentation model trained on the [TACO dataset](http://tacodataset.org/) using [Detectron2](https://github.com/facebookresearch/detectron2).

## Dataset

| Stat | Value |
|---|---|
| Total images | 193 |
| Total annotations | 481 |
| Classes | 62 (e.g. Aerosol, Glass bottle, Plastic film …) |
| Train split | 80% (~154 images) |
| Test split | 20% (~39 images) |

Annotations are in COCO format (`annotations.json`). Raw images live in `TACO_Raw/`.

> **Note:** ~8 annotations per class on average. Expect lower AP on rare classes. Collecting more data or merging similar categories will help.

## Setup

```bash
# Python 3.8+ recommended
pip install torch torchvision
pip install detectron2   # follow https://detectron2.readthedocs.io/en/latest/tutorials/install.html
pip install opencv-python
```

**Training on an NVIDIA GPU (recommended):** see [SETUP_CUDA_WINDOWS.md](SETUP_CUDA_WINDOWS.md) for a full Windows + CUDA walkthrough. CPU training takes ~12 hours; a mid-range GPU finishes in 15–30 min.

## Dataset location

`training.py` and `evaluation.py` read the TACO dataset path from the `TACO_DATA_DIR` environment variable, falling back to a hardcoded Mac path. Set it before running on a new machine:

```bash
# macOS / Linux
export TACO_DATA_DIR=/path/to/TACO_repo/data

# Windows PowerShell
$env:TACO_DATA_DIR = "C:\path\to\TACO_repo\data"
```

## Training

```bash
python training.py
```

Checkpoints are written to `output/` every 500 iterations. Final weights: `output/model_final.pth`.

## Evaluation

COCO-style AP / AR metrics are printed automatically after training via `COCOEvaluator`.

## Project Structure

```
.
├── annotations.json       # COCO-format annotations
├── TACO_Raw/              # Raw images
├── training.py            # Training + evaluation + visualisation
├── evaluation.py          # Standalone evaluation script
├── confusion_matrix.py    # Confusion matrix helper
└── output/                # Checkpoints and metrics (generated)
```

---

## Training with Custom Images

### 1. Supported Image Formats

| Format | Supported |
|---|---|
| `.jpg` / `.jpeg` | Yes (recommended) |
| `.png` | Yes |
| `.bmp` | Yes |
| `.tiff` / `.tif` | Yes |

JPEG is recommended for photos (smaller file size). PNG is better when lossless quality matters.

### 2. Annotate Your Images

Your `training.py` uses COCO-format annotations. Use one of these tools to annotate and export:

- **[CVAT](https://github.com/cvat-ai/cvat)** — free, self-hostable, exports COCO JSON
- **[Label Studio](https://labelstud.io/)** — free, web UI, exports COCO JSON
- **[Roboflow](https://roboflow.com/)** — web UI, free tier available, exports COCO JSON

Export your annotations as a single COCO-format `.json` file.

### 3. Folder Structure

```
my_project/
├── my_images/             # all your images go here
├── my_annotations.json    # COCO-format annotation file
└── training.py
```

### 4. Update `training.py`

Change line 23 to point to your images folder and annotation file:

```python
register_coco_instances("custom_dataset", {}, "my_annotations.json", "my_images/")
```

### 5. Run Training

```bash
python training.py
```

The model will train on your data and save the final weights to `output/model_final.pth`.

### Tips

- Aim for at least **50–100 images per class** for reasonable accuracy.
- More classes = more data needed. If you have few images, consider merging similar categories.
- If retraining from scratch, clear the old output first:
  ```bash
  rm -rf output/ && python training.py
  ```
- To resume a stopped training run, set `resume=True` in `training.py` line 74:
  ```python
  trainer.resume_or_load(resume=True)
  ```

---

## Changes & Optimisations (April 2026)

### Critical Fixes

| Setting | Before | After | Reason |
|---|---|---|---|
| `SOLVER.MAX_ITER` | `100` | `3000` | 100 iters ≈ 2.6 epochs — model barely trains |
| `ROI_HEADS.BATCH_SIZE_PER_IMAGE` | `16` | `128` | 16 proposals starves the 62-class classifier of negatives |

### Solver Improvements

| Setting | Before | After | Reason |
|---|---|---|---|
| `SOLVER.WARMUP_ITERS` | not set | `200` | Prevents LR instability in early iterations |
| `SOLVER.STEPS` | not set | `(2000, 2500)` | LR decays at 2/3 and 5/6 of training |
| `SOLVER.GAMMA` | not set | `0.1` | Multiply LR by 0.1 at each step |
| `SOLVER.CHECKPOINT_PERIOD` | not set | `500` | Save checkpoint every 500 iters so a crash doesn't lose all progress |

### Data Augmentation

Added `AugmentedTrainer` subclass of `DefaultTrainer` with:
- Horizontal flip
- Random brightness ±20%
- Random contrast ±20%
- Multi-scale resize (640–800 px short edge)

Critical for a small dataset to prevent overfitting.

### Evaluation Re-enabled

`COCOEvaluator` + `inference_on_dataset` were commented out. Now runs automatically after training and prints AP/AR breakdown per class.

### Reproducibility

Added `random.seed(42)` before `random.shuffle(dataset)` so the train/test split is identical across runs.

### Visualisation Threshold

`SCORE_THRESH_TEST` raised from `0.1` → `0.5` to suppress low-confidence false-positive detections in the preview window.
