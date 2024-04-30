import os
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances


def run():
    # Path to your COCO format annotations file and image directory
    json_annotations_file = "annotations.json" # path to annotation files
    image_root = "TACO_Raw" # Directory of the images

    # Register the COCO format dataset
    register_coco_instances("custom_dataset", {}, json_annotations_file, image_root)

    # Load annotations and extract class names
    with open(json_annotations_file, "r") as f:
        annotations = json.load(f)
    class_names = [category["name"] for category in annotations["categories"]]

    # Optionally, set metadata for your dataset
    MetadataCatalog.get("custom_dataset").set(thing_classes=class_names)

    # Define dataset and model configurations
    dataset_name = "custom_dataset"
    model_weights_file = "model(8_10000).pth" #path to the weights file

    # Load the pre-trained model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set detection threshold value

    # Step 6: Initialize the evaluator
    evaluator = COCOEvaluator("custom_dataset", cfg, False, output_dir="./output1/") # output directory

    # Step 7: Run evaluation
    predictor = DefaultPredictor(cfg)
    test_loader = build_detection_test_loader(cfg, "custom_dataset")
    inference_on_dataset(predictor.model, test_loader, evaluator)

if __name__ == "__main__":
    run()
