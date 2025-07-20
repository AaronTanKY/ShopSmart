from PIL import Image
import imagehash
from pathlib import Path

def compute_phash(image: str) -> str:
    """
    Compute the perceptual hash (pHash) of an image.

    Args:
        image_path: Path to the input image.

    Returns:
        pHash as a string.
    """
    # image = Image.open(image_path)
    phash = imagehash.phash(image)
    return str(phash)

# Optional: test standalone
if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent.resolve()
    image_path= CURRENT_DIR / "Trial/test/images/IMG-20250718-WA0037_jpg.rf.6ca7481275757de260af6a3e80ac7957.jpg",

    phash = compute_phash(image_path)
    print(f"pHash value: {phash}.")
    print("✅ pHash value printed.")
