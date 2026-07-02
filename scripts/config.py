from pathlib import Path

PROJECT_ROOT = Path.home() / "FishProject"

DATASET_PATH = PROJECT_ROOT / "datasets" / "Phase4_YOLO_Dataset_v1"

DATA_YAML = DATASET_PATH / "data.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_NAME = "yolo11s.pt"

IMAGE_SIZE = 640

EPOCHS = 100

BATCH_SIZE = 16

WORKERS = 8

DEVICE = 0

EXPERIMENT_NAME = "YOLO11_Fish_Detection"
