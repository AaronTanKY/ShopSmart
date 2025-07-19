from pathlib import Path
from ultralytics import YOLO

# Find current working directory
CURRENT_DIR = Path(__file__).parent.resolve()

# Path to the best trained weights
best_weights_path = CURRENT_DIR / "Trial" / "runs" / "experiment" / "weights" / "best.pt"

# Load best trained weights
model = YOLO(best_weights_path) 

# Perform inference on a test folder
test_dir = CURRENT_DIR / "Trial" / "test" / "images" # path to your test images
results = model(test_dir)  # batch inference

for r in results:
    r.show()   # show each image