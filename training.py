import os
import random
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
                        [640, 672, 704, 736, 768, 800], max_size=1333
                    ),
                ],
            ),
        )


def main():
    # Fix seed for reproducible train/test split
    random.seed(42)

    # Register the full COCO-format dataset
    register_coco_instances("custom_dataset", {}, "annotations.json", "TACO_Raw")

    dataset = DatasetCatalog.get("custom_dataset")
    random.shuffle(dataset)

    split_idx = int(len(dataset) * 0.8)  # 80/20 train/test
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    # Extract class names and count from the annotation file
    with open("annotations.json", "r") as f:
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
    trainer.train()

    # Save final checkpoint explicitly
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save("model_final")

    # --------------------------------------------------------------- evaluation
    evaluator = COCOEvaluator("test_dataset", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "test_dataset")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    # ------------------------------------------------------------ visualisation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # was 0.1 — too noisy
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get("train_dataset")

    for d in random.sample(
        DatasetCatalog.get("test_dataset"), min(30, len(test_dataset))
    ):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.25)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("prediction", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
