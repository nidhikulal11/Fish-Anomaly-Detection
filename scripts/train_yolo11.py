from ultralytics import YOLO
from config import *

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Fish Behaviour Project")
print("Phase 4 : YOLO11 Production Training")
print("=" * 60)

model = YOLO(MODEL_NAME)

results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    workers=WORKERS,
    device=DEVICE,
    project=str(OUTPUT_DIR),
    name=EXPERIMENT_NAME,
    pretrained=True,
    save=True,
    save_period=10,
    verbose=True,
    optimizer="auto",
    patience=20
)

print("\nTraining Finished Successfully")
