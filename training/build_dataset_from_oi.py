import os
import csv
import shutil
from pathlib import Path

RAW_DIR = "../data/raw/"
LABELS_FILE = "../data/labels.csv"

def main():
    person_dir = Path(RAW_DIR) / "person"
    no_person_dir = Path(RAW_DIR) / "no_person"

    final_dir = Path(RAW_DIR)
    rows = []
    index = 0

    for label, source in [("person", person_dir), ("no_person", no_person_dir)]:
        for img_file in source.rglob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            new_name = f"img_{index:05d}{img_file.suffix.lower()}"
            shutil.copy2(img_file, final_dir / new_name)
            rows.append((new_name, label))
            index += 1

    with open(LABELS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print("[INFO] Dataset built successfully.")
    print("[INFO] labels.csv created with", len(rows), "entries.")

if __name__ == "__main__":
    main()
