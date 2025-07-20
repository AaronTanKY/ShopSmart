from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

def get_best_crop(
    model_path: str,
    image_path: str,
    save_path: str = None,
) -> np.ndarray:
    """
    Run YOLO detection on an image and return the best crop (closest to center & largest area).

    Args:
        model_path: path to YOLO weights (.pt)
        image_path: path to input image
        save_path: optional path to save cropped image (e.g., 'output.jpg')

    Returns:
        Cropped image as numpy array (BGR)
    """
    model = YOLO(model_path)

    # Read original image
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    image_center = np.array([W / 2, H / 2])

    # Run inference
    results = model(image_path)

    # Find best box
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = result.boxes.conf.cpu().numpy()  # confidences
        cls = result.boxes.cls.cpu().numpy()     # class ids
        names = result.names

        if len(boxes) == 0:
            raise ValueError(f"No bounding boxes found in {image_path}")

        box_infos = []
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls)):
            x1, y1, x2, y2 = box
            box_center = np.array([(x1+x2)/2, (y1+y2)/2])
            dist_to_center = np.linalg.norm(box_center - image_center)
            area = (x2 - x1) * (y2 - y1)
            box_infos.append({
                'index': i,
                'box': box,
                'conf': conf,
                'cls_id': int(cls_id),
                'dist_to_center': dist_to_center,
                'area': area
            })

        # Sort by closest to center & largest area
        box_infos.sort(key=lambda b: (b['dist_to_center'], -b['area']))
        best = box_infos[0]

        x1, y1, x2, y2 = map(int, best['box'])
        crop = img[y1:y2, x1:x2]

        if save_path:
            cv2.imwrite(save_path, crop)

        return crop

    raise RuntimeError("Unexpected error: no results processed.")

# Optional test
if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent.resolve()
    crop = get_best_crop(
        model_path= CURRENT_DIR / "Trial/runs/experiment2/weights/best.pt",
        image_path= CURRENT_DIR / "Trial/test/images/IMG-20250718-WA0037_jpg.rf.6ca7481275757de260af6a3e80ac7957.jpg",
        save_path= CURRENT_DIR / "Photos/results/best_crop.jpg"
    )
    print("✅ Cropped image saved and returned.")
