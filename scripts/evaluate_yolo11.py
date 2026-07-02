from ultralytics import YOLO
from config import *

MODEL = (
    OUTPUT_DIR
    / EXPERIMENT_NAME
    / "weights"
    / "best.pt"
)

model = YOLO(str(MODEL))

metrics = model.val(data=str(DATA_YAML))

print(metrics)
