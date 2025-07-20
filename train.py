from pathlib import Path
from ultralytics import YOLO

# Find current working directory
CURRENT_DIR = Path(__file__).parent.resolve()

# # Load a pretrained model (you can choose nano, small, medium, etc.)
model = YOLO(CURRENT_DIR / "Trial" / "runs" / "experiment" / "weights" / "best.pt")

# Train on your dataset
results = model.train(
    data=CURRENT_DIR / "Trial" / "data.yaml",  # the Roboflow-generated YAML file
    epochs=100,
    imgsz=640,
    batch=16,
    project= CURRENT_DIR / "Trial" / "runs",  # path to save the results
    name="experiment2",  # name of the experiment
)

# Save the model or evaluate it further if needed
