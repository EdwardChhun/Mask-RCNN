"""Build a per-instance confusion matrix from COCO predictions vs ground truth.

Matches each ground-truth annotation to the highest-IoU prediction in the same
image (IoU >= 0.5). Unmatched GT instances are counted as predicted-as-background;
unmatched predictions are counted as false positives against background. The
result is saved as a normalized heatmap PNG.
"""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


IOU_THRESHOLD = 0.5


def _bbox_iou(a, b):
    """COCO bbox format: [x, y, w, h]. Returns IoU."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def build_confusion_matrix(predictions_json, annotations_json, output_path):
    with open(annotations_json, "r") as f:
        ann_data = json.load(f)
    with open(predictions_json, "r") as f:
        preds = json.load(f)

    classes = [c["name"] for c in ann_data["categories"]]
    # category_id 1..N maps to row/col 0..N-1; last row/col is "background"
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(ann_data["categories"])}
    n = len(classes)
    bg = n  # background index
    cm = np.zeros((n + 1, n + 1), dtype=np.int64)

    gts_by_img = defaultdict(list)
    for ann in ann_data["annotations"]:
        gts_by_img[ann["image_id"]].append(ann)

    preds_by_img = defaultdict(list)
    for p in preds:
        preds_by_img[p["image_id"]].append(p)

    for image_id, gts in gts_by_img.items():
        ps = preds_by_img.get(image_id, [])
        # Greedy 1-to-1 match: for each GT take the best-IoU unmatched prediction.
        used_pred = set()
        gt_matched = [False] * len(gts)
        # Sort GTs by area desc so big objects get first dibs on predictions.
        gt_order = sorted(range(len(gts)), key=lambda i: -gts[i].get("area", 0))
        for gi in gt_order:
            gt = gts[gi]
            gt_idx = cat_id_to_idx.get(gt["category_id"])
            if gt_idx is None:
                continue
            best_iou, best_pi = 0.0, -1
            for pi, p in enumerate(ps):
                if pi in used_pred:
                    continue
                iou = _bbox_iou(gt["bbox"], p["bbox"])
                if iou > best_iou:
                    best_iou, best_pi = iou, pi
            if best_pi >= 0 and best_iou >= IOU_THRESHOLD:
                pred_idx = cat_id_to_idx.get(ps[best_pi]["category_id"], bg)
                cm[gt_idx, pred_idx] += 1
                used_pred.add(best_pi)
                gt_matched[gi] = True
            else:
                cm[gt_idx, bg] += 1   # missed detection
        # Remaining unmatched predictions are false positives.
        for pi, p in enumerate(ps):
            if pi in used_pred:
                continue
            pred_idx = cat_id_to_idx.get(p["category_id"], bg)
            cm[bg, pred_idx] += 1

    labels = classes + ["(background)"]
    _plot_and_save(cm, labels, output_path)
    return cm


def _plot_and_save(cm, labels, output_path, fontsize=5, figsize=(12, 10)):
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Normalized confusion matrix (IoU >= {IOU_THRESHOLD})")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xlabel("Predicted", fontsize=fontsize + 2)
    ax.set_ylabel("True", fontsize=fontsize + 2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="output/coco_instances_results.json")
    p.add_argument("--annotations", default="annotations_taco_fixed.json")
    p.add_argument("--output", default="output/report/confusion_matrix.png")
    args = p.parse_args()
    build_confusion_matrix(args.predictions, args.annotations, args.output)
    print(f"Wrote {args.output}")
