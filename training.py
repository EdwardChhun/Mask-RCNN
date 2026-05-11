import os
import random
import sys
import time
import cv2
import torch
import torchvision
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import (
    DatasetCatalog, MetadataCatalog,
    build_detection_test_loader, build_detection_train_loader,
    DatasetMapper,
)
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
import json

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())


class AugmentedTrainer(DefaultTrainer):
    """DefaultTrainer with multi-scale + photometric augmentations."""

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=[
                    T.RandomFlip(horizontal=True),
                    T.RandomBrightness(0.8, 1.2),
                    T.RandomContrast(0.8, 1.2),
                    T.ResizeShortestEdge(
                        [640, 672, 704, 736, 768, 800],
                        max_size=1333,
                        sample_style="choice",
                    ),
                ],
            ),
        )


TACO_DATA_DIR = os.environ.get(
    "TACO_DATA_DIR",
    r"C:\Developer\TACO_repo\data",
)
TACO_ANNOTATIONS_SRC = os.path.join(TACO_DATA_DIR, "annotations.json")
TACO_ANNOTATIONS_FIXED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "annotations_taco_fixed.json"
)


def fix_taco_annotations(src_path, dst_path):
    """TACO's annotations.json has duplicate annotation ids and category ids
    that aren't contiguous from 1..N. Detectron2's COCO loader rejects the
    duplicates outright. Reassign both id sets and write a fixed copy."""
    with open(src_path, "r") as f:
        data = json.load(f)

    # Remap category ids to 1..N (preserve relative order from source)
    cat_id_map = {}
    for new_id, cat in enumerate(data["categories"], start=1):
        cat_id_map[cat["id"]] = new_id
        cat["id"] = new_id

    # Reassign annotation ids sequentially and remap category_id
    for new_id, ann in enumerate(data["annotations"], start=1):
        ann["id"] = new_id
        ann["category_id"] = cat_id_map[ann["category_id"]]

    with open(dst_path, "w") as f:
        json.dump(data, f)


def main():
    # Fix seed for reproducible train/test split
    random.seed(42)

    # Rebuild the fixed annotations file if missing or stale
    if (not os.path.exists(TACO_ANNOTATIONS_FIXED)
            or os.path.getmtime(TACO_ANNOTATIONS_FIXED)
            < os.path.getmtime(TACO_ANNOTATIONS_SRC)):
        print(f"Writing deduped annotations to {TACO_ANNOTATIONS_FIXED}")
        fix_taco_annotations(TACO_ANNOTATIONS_SRC, TACO_ANNOTATIONS_FIXED)

    # Register the full COCO-format dataset
    register_coco_instances(
        "custom_dataset", {},
        TACO_ANNOTATIONS_FIXED,
        TACO_DATA_DIR,
    )

    dataset = DatasetCatalog.get("custom_dataset")

    # Filter to only images that were successfully downloaded
    dataset = [d for d in dataset if os.path.exists(d["file_name"])]
    print(f"Using {len(dataset)} images with files present on disk.")

    random.shuffle(dataset)

    split_idx = int(len(dataset) * 0.8)  # 80/20 train/test
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    # Extract class names and count from the fixed annotation file
    with open(TACO_ANNOTATIONS_FIXED, "r") as f:
        annotations = json.load(f)

    class_names = [category["name"] for category in annotations["categories"]]
    num_classes = len(class_names)

    # Register the splits
    DatasetCatalog.register("train_dataset", lambda: train_dataset)
    MetadataCatalog.get("train_dataset").set(thing_classes=class_names)
    DatasetCatalog.register("test_dataset", lambda: test_dataset)
    MetadataCatalog.get("test_dataset").set(thing_classes=class_names)

    # ------------------------------------------------------------------ config
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("test_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # Pretrained COCO weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Solver — was MAX_ITER=100, now 3000; was LR=0.001 with no schedule
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.WARMUP_ITERS = 200          # stabilise LR at start
    cfg.SOLVER.STEPS = (2000, 2500)        # decay LR at these iterations
    cfg.SOLVER.GAMMA = 0.1                 # multiply LR by 0.1 at each step
    cfg.SOLVER.CHECKPOINT_PERIOD = 500    # save every 500 iters

    # ROI heads — was BATCH_SIZE_PER_IMAGE=16 (too small for 62 classes)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------- training
    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    train_start = time.time()
    trainer.train()
    train_seconds = time.time() - train_start

    # Save final checkpoint explicitly
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save("model_final")

    # --------------------------------------------------------------- evaluation
    evaluator = COCOEvaluator("test_dataset", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "test_dataset")
    eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)

    # ----------------------------------------------------------- report assets
    report_dir = os.path.join(cfg.OUTPUT_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "coco_metrics.json"), "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    write_summary(
        os.path.join(report_dir, "summary.txt"),
        eval_results=eval_results,
        train_seconds=train_seconds,
        num_train=len(train_dataset),
        num_test=len(test_dataset),
        num_classes=num_classes,
        device=cfg.MODEL.DEVICE,
    )

    # Loss curves from detectron2's metrics.json (JSONL written during training)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "visualization"))
        from visualize import Visualizer as MetricsVisualizer
        mv = MetricsVisualizer(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))
        for name, fig in mv.iter_figures():
            fig.write_image(os.path.join(report_dir, f"{name}.png"))
    except Exception as e:
        print(f"[report] loss-curve generation skipped: {e}")

    # Confusion matrix (uses coco_instances_results.json written by evaluator)
    try:
        from confusion_matrix import build_confusion_matrix
        build_confusion_matrix(
            predictions_json=os.path.join(cfg.OUTPUT_DIR, "coco_instances_results.json"),
            annotations_json=TACO_ANNOTATIONS_FIXED,
            output_path=os.path.join(report_dir, "confusion_matrix.png"),
        )
    except Exception as e:
        print(f"[report] confusion matrix skipped: {e}")

    # ------------------------------------------------------------ visualisation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # was 0.1 — too noisy
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get("train_dataset")
    show_previews = bool(os.environ.get("SHOW_PREVIEWS"))
    preds_dir = os.path.join(report_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    sample = random.sample(
        DatasetCatalog.get("test_dataset"), min(10, len(test_dataset))
    )
    for i, d in enumerate(sample):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_bgr = v.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(preds_dir, f"{i:02d}.jpg"), out_bgr)
        if show_previews:
            cv2.imshow("prediction", out_bgr)
            cv2.waitKey(0)
    if show_previews:
        cv2.destroyAllWindows()

    print(f"\nReport written to {report_dir}")


def write_summary(path, *, eval_results, train_seconds, num_train,
                  num_test, num_classes, device):
    """Write a human-readable summary of training + evaluation."""
    def fmt_seconds(s):
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h:d}h {m:02d}m {sec:02d}s"

    lines = []
    lines.append("Mask R-CNN training summary")
    lines.append("=" * 40)
    lines.append(f"Device:           {device}")
    lines.append(f"Train images:     {num_train}")
    lines.append(f"Test images:      {num_test}")
    lines.append(f"Classes:          {num_classes}")
    lines.append(f"Training time:    {fmt_seconds(train_seconds)}")
    lines.append("")
    for task in ("bbox", "segm"):
        if task not in eval_results:
            continue
        m = eval_results[task]
        lines.append(f"[{task}]")
        for k in ("AP", "AP50", "AP75", "APs", "APm", "APl"):
            if k in m:
                lines.append(f"  {k:<6} {m[k]:.3f}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
