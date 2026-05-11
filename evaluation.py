import os
import json
import random
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances


TACO_DATA_DIR = os.environ.get(
    "TACO_DATA_DIR",
    "/Users/dr.chhunry/Desktop/Developer/TACO_repo/data",
)
TACO_ANNOTATIONS_FIXED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "annotations_taco_fixed.json"
)


def run():
    model_weights_file = "output/model_final.pth"
    output_dir = "./output1/"

    register_coco_instances(
        "custom_dataset", {},
        TACO_ANNOTATIONS_FIXED,
        TACO_DATA_DIR,
    )

    with open(TACO_ANNOTATIONS_FIXED, "r") as f:
        annotations = json.load(f)
    class_names = [category["name"] for category in annotations["categories"]]
    MetadataCatalog.get("custom_dataset").set(thing_classes=class_names)

    # Mirror training.py: filter to images present on disk BEFORE the seeded shuffle
    # so we land on the same 80/20 split and evaluate only on the held-out 20%.
    random.seed(42)
    dataset = DatasetCatalog.get("custom_dataset")
    dataset = [d for d in dataset if os.path.exists(d["file_name"])]
    print(f"Using {len(dataset)} images with files present on disk.")
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    test_dataset = dataset[split_idx:]

    DatasetCatalog.register("test_dataset", lambda: test_dataset)
    MetadataCatalog.get("test_dataset").set(thing_classes=class_names)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("test_dataset",)
    cfg.DATASETS.TEST = ("test_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    evaluator = COCOEvaluator("test_dataset", cfg, False, output_dir=output_dir)
    predictor = DefaultPredictor(cfg)
    test_loader = build_detection_test_loader(cfg, "test_dataset")
    inference_on_dataset(predictor.model, test_loader, evaluator)


if __name__ == "__main__":
    run()
