# training/download_openimages_dataset.py

import os
import csv
import shutil
from pathlib import Path

from tqdm import tqdm
from openimages.download import download_images

# Where to temporarily download Open Images structure
TMP_DIR = "../data/openimages_tmp"
# Where EdgeGuard_AI expects training images
RAW_DIR = "../data/raw"
LABELS_FILE = "../data/labels.csv"

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# Open Images class names (must match official class names)
# See: Open Images V4 class list (e.g., Person, Man, Woman, Girl, Boy, Tree, Car, Building, Cat, Dog, Flower) :contentReference[oaicite:1]{index=1}
PERSON_CLASSES = ["Person", "Man", "Woman", "Girl", "Boy"]
NO_PERSON_CLASSES = ["Tree", "Car", "Building", "Cat", "Dog", "Flower"]

# How many images per class to attempt to download
PERSON_LIMIT_PER_CLASS = 80   # 5 classes * 80 ≈ 400 person images
NO_PERSON_LIMIT_PER_CLASS = 80  # 6 classes * 80 ≈ 480 no_person images


def download_class_images(class_names, limit_per_class):
    """
    Use openimages.download.download_images to grab images for each class.
    Images will be stored under TMP_DIR/<ClassName>/images/*.jpg
    """
    for cls in class_names:
        print(f"[INFO] Downloading class '{cls}' (limit={limit_per_class})...")
        # This creates TMP_DIR/<cls>/images/...
        download_images(
            TMP_DIR,
            [cls],
            limit=limit_per_class
        )


def build_edgeguard_dataset():
    """
    Walks through TMP_DIR/<Class>/images, copies images into RAW_DIR,
    and creates labels.csv with labels 'person' / 'no_person'.
    """
    rows = []
    index = 0

    # Helper to copy images and assign unified label
    def process_class_dir(class_name, label):
        nonlocal index, rows

        images_dir = Path(TMP_DIR) / class_name / "images"
        if not images_dir.exists():
            print(f"[WARN] No images directory for class '{class_name}' at {images_dir}")
            return

        img_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        print(f"[INFO] Found {len(img_files)} images for class '{class_name}'")

        for img_path in tqdm(img_files, desc=f"Copying {class_name} as {label}"):
            filename = f"img_{index:05d}{img_path.suffix.lower()}"
            dest_path = Path(RAW_DIR) / filename

            # Copy file
            shutil.copy2(img_path, dest_path)

            # Add row to labels list
            rows.append((filename, label))
            index += 1

    # Positive (person) classes
    for cls in PERSON_CLASSES:
        process_class_dir(cls, "person")

    # Negative (no_person) classes
    for cls in NO_PERSON_CLASSES:
        process_class_dir(cls, "no_person")

    # Write labels.csv
    with open(LABELS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} labeled samples to {LABELS_FILE}")
    print(f"[INFO] Raw images stored in {RAW_DIR}")


def main():
    print("[INFO] Starting Open Images download for EdgeGuard_AI...")

    print("[INFO] Downloading PERSON images...")
    download_class_images(PERSON_CLASSES, PERSON_LIMIT_PER_CLASS)

    print("[INFO] Downloading NO_PERSON images...")
    download_class_images(NO_PERSON_CLASSES, NO_PERSON_LIMIT_PER_CLASS)

    print("[INFO] Building EdgeGuard_AI dataset structure...")
    build_edgeguard_dataset()

    print("[INFO] Dataset build complete.")
    print("[INFO] You can now re-run: python3 train_model.py")


if __name__ == "__main__":
    main()
