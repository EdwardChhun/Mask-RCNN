"""Evaluate a trained Mask R-CNN checkpoint on the TACO test split.

Produces, in EVAL_DIR:
  - coco_metrics.json           COCO AP/AR (bbox + segm), overall + per-class
  - per_class_metrics.csv       Per-class precision/recall/F1/support (IoU 0.5)
  - confusion_matrix_counts.csv Raw matched counts (N+1 x N+1, last row/col = background)
  - confusion_matrix.png        Normalized confusion matrix over all classes
  - confusion_matrix_top20.png  Same, restricted to the 20 most-supported classes
  - per_class_ap.png            Per-class segm AP bar chart
"""

import csv
import json
import os
import random

import matplotlib

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import seaborn as sns
import torch
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
TACO_DATA_DIR = os.environ.get(
    "TACO_DATA_DIR",
    "/Users/dr.chhunry/Desktop/Developer/TACO_repo/data",
)
ANNOTATIONS_FIXED = os.path.join(ROOT, "annotations_taco_fixed.json")
OUTPUT_DIR = os.path.join(ROOT, "output")
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "model_final.pth")
EVAL_DIR = os.path.join(ROOT, "eval")

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5


def build_test_split():
    """Reproduce the seed=42 80/20 split used by training.py."""
    random.seed(42)

    register_coco_instances(
        "custom_dataset", {}, ANNOTATIONS_FIXED, TACO_DATA_DIR
    )
    dataset = DatasetCatalog.get("custom_dataset")
    dataset = [d for d in dataset if os.path.exists(d["file_name"])]
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    test_dataset = dataset[split_idx:]

    with open(ANNOTATIONS_FIXED, "r") as f:
        annotations = json.load(f)
    class_names = [c["name"] for c in annotations["categories"]]

    DatasetCatalog.register("test_dataset", lambda: test_dataset)
    MetadataCatalog.get("test_dataset").set(thing_classes=class_names)
    return test_dataset, class_names


def build_cfg(num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ("test_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = OUTPUT_DIR
    return cfg


def run_coco_evaluator(cfg):
    evaluator = COCOEvaluator(
        "test_dataset", output_dir=EVAL_DIR, tasks=("bbox", "segm")
    )
    test_loader = build_detection_test_loader(cfg, "test_dataset")
    # Build the predictor model from cfg so the same code path is used.
    predictor = DefaultPredictor(cfg)
    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    return results, predictor


def coco_polys_to_mask(segm, h, w):
    """Convert a COCO segmentation (polygons or RLE) to a binary mask."""
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm, dict):
        if isinstance(segm.get("counts"), list):
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            rle = segm
    else:
        return None
    return mask_utils.decode(rle).astype(bool)


def load_gt_instances(test_dataset):
    """Return {image_id: [(class_idx, mask), ...]} for the test split."""
    gt = {}
    for d in test_dataset:
        h, w = d["height"], d["width"]
        items = []
        for ann in d["annotations"]:
            segm = ann.get("segmentation")
            if segm is None:
                continue
            mask = coco_polys_to_mask(segm, h, w)
            if mask is None or mask.sum() == 0:
                continue
            items.append((ann["category_id"], mask))
        gt[d["image_id"]] = items
    return gt


def predict_instances(predictor, test_dataset):
    """Run the predictor over every test image. Returns
    {image_id: [(class_idx, score, mask), ...]} sorted by score desc."""
    import cv2

    preds = {}
    for d in tqdm(test_dataset, desc="predicting"):
        im = cv2.imread(d["file_name"])
        if im is None:
            preds[d["image_id"]] = []
            continue
        out = predictor(im)["instances"].to("cpu")
        if len(out) == 0:
            preds[d["image_id"]] = []
            continue
        masks = out.pred_masks.numpy().astype(bool)
        classes = out.pred_classes.numpy().tolist()
        scores = out.scores.numpy().tolist()
        items = list(zip(classes, scores, masks))
        items.sort(key=lambda x: x[1], reverse=True)
        preds[d["image_id"]] = items
    return preds


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def build_confusion_matrix(gt, preds, num_classes, iou_thresh=IOU_THRESHOLD):
    """Greedy IoU matching per image. Returns an (N+1)x(N+1) matrix where the
    last row/col is 'background' (unmatched GT / unmatched prediction)."""
    bg = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for image_id, gt_items in gt.items():
        pred_items = preds.get(image_id, [])
        gt_taken = [False] * len(gt_items)

        # Greedy: predictions are pre-sorted by score; pick the best-IoU GT.
        for p_cls, _score, p_mask in pred_items:
            best_iou = 0.0
            best_idx = -1
            for i, (_g_cls, g_mask) in enumerate(gt_items):
                if gt_taken[i]:
                    continue
                iou = mask_iou(p_mask, g_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= iou_thresh:
                gt_taken[best_idx] = True
                g_cls = gt_items[best_idx][0]
                cm[g_cls, p_cls] += 1
            else:
                cm[bg, p_cls] += 1  # false positive

        for i, (g_cls, _g_mask) in enumerate(gt_items):
            if not gt_taken[i]:
                cm[g_cls, bg] += 1  # false negative

    return cm


def per_class_prf(cm, num_classes):
    """precision/recall/F1/support per class from confusion matrix."""
    rows = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)              # row c minus diagonal
        fp = int(cm[:, c].sum() - tp)              # col c minus diagonal
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        rows.append(
            {
                "class_id": c,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    return rows


def plot_confusion(cm, class_names, out_path, title):
    """Plot a row-normalized confusion matrix as a heatmap."""
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig_w = max(8, len(class_names) * 0.25)
    plt.figure(figsize=(fig_w, fig_w * 0.85))
    sns.heatmap(
        norm,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=False,
        cbar_kws={"label": "fraction of GT row"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_class_ap(coco_results, class_names, out_path):
    """Bar chart of per-class segm AP if available."""
    segm = coco_results.get("segm", {})
    per_cls = {k: v for k, v in segm.items() if k.startswith("AP-")}
    if not per_cls:
        return
    pairs = []
    for k, v in per_cls.items():
        name = k[3:]
        try:
            pairs.append((name, float(v)))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*pairs)
    plt.figure(figsize=(max(8, len(names) * 0.25), 6))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
    plt.ylabel("segm AP")
    plt.title("Per-class segm AP (TACO test split)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if not os.path.exists(WEIGHTS_PATH):
        raise SystemExit(
            f"Weights not found at {WEIGHTS_PATH}. "
            "Run `python training.py` first."
        )

    os.makedirs(EVAL_DIR, exist_ok=True)
    test_dataset, class_names = build_test_split()
    num_classes = len(class_names)
    print(f"Test images: {len(test_dataset)} | classes: {num_classes}")

    cfg = build_cfg(num_classes)
    coco_results, predictor = run_coco_evaluator(cfg)

    with open(os.path.join(EVAL_DIR, "coco_metrics.json"), "w") as f:
        json.dump(coco_results, f, indent=2, default=str)

    plot_per_class_ap(
        coco_results, class_names, os.path.join(EVAL_DIR, "per_class_ap.png")
    )

    # ---- instance-level confusion matrix ---------------------------------
    print("Loading ground-truth masks…")
    gt = load_gt_instances(test_dataset)
    print("Running predictor for confusion matrix…")
    preds = predict_instances(predictor, test_dataset)

    cm = build_confusion_matrix(gt, preds, num_classes)
    labels = class_names + ["__background__"]

    counts_path = os.path.join(EVAL_DIR, "confusion_matrix_counts.csv")
    with open(counts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, row_label in enumerate(labels):
            writer.writerow([row_label] + cm[i].tolist())

    plot_confusion(
        cm,
        labels,
        os.path.join(EVAL_DIR, "confusion_matrix.png"),
        f"Normalized confusion matrix (IoU ≥ {IOU_THRESHOLD})",
    )

    # Top-20-by-support variant for readability
    supports = cm[:num_classes, :].sum(axis=1)  # GT counts per class
    top_idx = np.argsort(-supports)[:20]
    keep = list(top_idx) + [num_classes]  # keep background row/col
    cm_top = cm[np.ix_(keep, keep)]
    top_labels = [labels[i] for i in keep]
    plot_confusion(
        cm_top,
        top_labels,
        os.path.join(EVAL_DIR, "confusion_matrix_top20.png"),
        "Confusion matrix — top 20 classes by GT support",
    )

    # ---- per-class precision/recall/F1 -----------------------------------
    rows = per_class_prf(cm, num_classes)
    prf_path = os.path.join(EVAL_DIR, "per_class_metrics.csv")
    with open(prf_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "tp",
                "fp",
                "fn",
                "support",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        for r in rows:
            r["class_name"] = class_names[r["class_id"]]
            writer.writerow(r)

    # ---- console summary --------------------------------------------------
    bbox = coco_results.get("bbox", {})
    segm = coco_results.get("segm", {})
    print("\n=== COCO summary (test split) ===")
    for task, d in (("bbox", bbox), ("segm", segm)):
        ap = d.get("AP")
        ap50 = d.get("AP50")
        ap75 = d.get("AP75")
        print(f"{task}: AP={ap}  AP50={ap50}  AP75={ap75}")

    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r)
        else 0.0
    )
    print(
        f"\nMicro precision={micro_p:.3f} recall={micro_r:.3f} "
        f"F1={micro_f1:.3f}  (TP={total_tp} FP={total_fp} FN={total_fn})"
    )
    print(f"\nArtifacts written to {EVAL_DIR}/")


if __name__ == "__main__":
    main()
