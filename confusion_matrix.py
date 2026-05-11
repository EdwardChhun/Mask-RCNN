"""Replot the confusion matrix that evaluation.py has already saved.

The heavy lifting (IoU-matched instance assignment, counts) lives in
evaluation.py — this script just reloads eval/confusion_matrix_counts.csv
and re-renders the PNG with whatever knobs you want to tweak.

Usage:
    python confusion_matrix.py              # full 63x63 plot
    python confusion_matrix.py --top 20     # only the 20 most-supported classes
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(ROOT, "eval")
COUNTS_CSV = os.path.join(EVAL_DIR, "confusion_matrix_counts.csv")


def load_counts(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    labels = rows[0][1:]
    cm = np.array([[int(x) for x in r[1:]] for r in rows[1:]], dtype=np.int64)
    return cm, labels


def plot(cm, labels, out_path, title):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    side = max(8, len(labels) * 0.25)
    plt.figure(figsize=(side, side * 0.85))
    sns.heatmap(
        norm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
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
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="If >0, restrict plot to the N most-supported true classes.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: eval/confusion_matrix[_topN].png)",
    )
    args = parser.parse_args()

    if not os.path.exists(COUNTS_CSV):
        raise SystemExit(
            f"Missing {COUNTS_CSV}. Run `python evaluation.py` first."
        )

    cm, labels = load_counts(COUNTS_CSV)

    if args.top > 0:
        # Last row/col is __background__; rank true classes by GT support.
        gt_supports = cm[:-1, :].sum(axis=1)
        top_idx = np.argsort(-gt_supports)[: args.top]
        keep = list(top_idx) + [len(labels) - 1]
        cm = cm[np.ix_(keep, keep)]
        labels = [labels[i] for i in keep]
        out_path = args.out or os.path.join(
            EVAL_DIR, f"confusion_matrix_top{args.top}.png"
        )
        title = f"Confusion matrix — top {args.top} classes"
    else:
        out_path = args.out or os.path.join(EVAL_DIR, "confusion_matrix.png")
        title = "Normalized confusion matrix"

    plot(cm, labels, out_path, title)


if __name__ == "__main__":
    main()
