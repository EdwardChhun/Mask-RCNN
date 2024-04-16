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
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import json

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())


def main():
    # Register the COCO format dataset
    register_coco_instances("custom_dataset", {}, "annotations.json", "TACO_Raw")

    # Load the dataset
    dataset = DatasetCatalog.get("custom_dataset")

    # Shuffle the dataset
    random.shuffle(dataset)

    # Calculate the split index
    split_idx = int(len(dataset) * 0.8)  # 80% for training, 20% for testing

    # Split the dataset into train and test
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]


    # Load annotations and extract class names
    with open("annotations.json", "r") as f:
        annotations = json.load(f)

    # Extract class names from annotations
    class_names = [category["name"] for category in annotations["categories"]]

    # Determine the number of classes
    num_classes = len(class_names)

    # Register train and test datasets
    DatasetCatalog.register("train_dataset", lambda: train_dataset)
    MetadataCatalog.get("train_dataset").set(thing_classes=class_names)  # Assuming class names are available
    DatasetCatalog.register("test_dataset", lambda: test_dataset)
    MetadataCatalog.get("test_dataset").set(thing_classes=class_names)  # Assuming class names are available

    # Define the configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("test_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 100
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create a trainer instance and start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    #model_file = os.path.join(cfg.OUTPUT_DIR, "model.pth")

    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save("model_final2")

    #trainer. save_model(model_file)

    # After training, perform evaluation
    #evaluator = COCOEvaluator("test_dataset", output_dir=cfg.OUTPUT_DIR)
    #trainer.test(cfg, trainer.model, evaluators=[evaluator])
    #evaluator = COCOEvaluator("test_dataset", cfg, False, output_dir=cfg.OUTPUT_DIR)
    #val_loader = build_detection_test_loader(cfg, "test_dataset")
    #inference_on_dataset(trainer.model, val_loader, evaluator)

    # Visualize predictions (optional)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get("train_dataset")

    # Randomly select 30 images from the validation dataset for visualization
    for d in random.sample(DatasetCatalog.get("test_dataset"), 30):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.25)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imshow("prediction", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
