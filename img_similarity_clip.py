import torch
import clip
from PIL import Image
import argparse
from pathlib import Path

def compute_similarity(image_path1, image_path2):
    device = "cpu"  # force CPU since you don't have CUDA

    print("Loading CLIP model on CPU…")
    model, preprocess = clip.load("ViT-B/32", device=device)

    print(f"Loading and preprocessing images…")
    image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding1 = model.encode_image(image1)
        embedding2 = model.encode_image(image2)

    # Normalize embeddings
    embedding1 /= embedding1.norm()
    embedding2 /= embedding2.norm()

    similarity = (embedding1 @ embedding2.T).item()
    return similarity

if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent.resolve()

    image1_dir = CURRENT_DIR / "Photos" / "image2.jpg"
    image2_dir = CURRENT_DIR / "Photos" / "image2.jpg"

    similarity = compute_similarity(image1_dir, image2_dir)
    print(f"\n✅ Similarity score: {similarity:.4f} (1.0 = identical, 0 = unrelated)")

    if similarity > 0.85:
        print("🔷 Images are likely the same or very similar.")
    elif similarity > 0.5:
        print("🔷 Images are somewhat similar.")
    else:
        print("🔷 Images are likely different.")
