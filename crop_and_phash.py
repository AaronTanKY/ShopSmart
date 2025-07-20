from crop import get_best_crop
from phash import compute_phash
from PIL import Image
import cv2
from pathlib import Path

def crop_and_compute_phash(model_path: str, image_path: str, save_crop_path: str = None) -> str:
    """
    Combines cropping and pHash computation:
    - uses YOLO to crop the best bbox
    - computes pHash of the crop
    - optionally saves the crop

    Args:
        model_path: YOLO weights path
        image_path: input image path
        save_crop_path: optional output path to save crop

    Returns:
        pHash string
    """
    # get cropped image (numpy array)
    cropped_np = get_best_crop(model_path, image_path)

    # convert numpy (BGR) to PIL (RGB)
    cropped_rgb = cv2.cvtColor(cropped_np, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)

    if save_crop_path:
        cropped_pil.save(save_crop_path)

    # compute pHash
    phash = compute_phash(cropped_pil)

    return phash


if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent.resolve()
    model_path= CURRENT_DIR / "Trial/runs/experiment2/weights/best.pt"
    image_path= CURRENT_DIR / "Trial/test/images/IMG-20250718-WA0046_jpg.rf.45afdea2218271674cfd569fad4c34d1.jpg"
    save_path= CURRENT_DIR / "Photos/results/best_crop.jpg"

    phash = crop_and_compute_phash(str(model_path), str(image_path), str(save_path))
    print(f"pHash of cropped image: {phash}")
