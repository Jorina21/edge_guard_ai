import os
import csv
import requests
import random
import shutil
from tqdm import tqdm

SAVE_DIR = "../data/raw/"
LABELS_FILE = "../data/labels.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

# Public datasets for quick image sampling
PERSON_URLS = [
    "https://api.unsplash.com/photos/random?query=person&count=30&client_id=_7Np5eigQ2UZBn-Hy8afZNq7QBKqZCphbv6cs6JnB6g"
]

NO_PERSON_URLS = [
    "https://api.unsplash.com/photos/random?query=empty+room&count=30&client_id=_7Np5eigQ2UZBn-Hy8afZNq7QBKqZCphbv6cs6JnB6g"
]

def download_images(url_list, label, start_index):
    index = start_index
    rows = []

    for api_url in url_list:
        print(f"Fetching dataset from: {api_url}")

        response = requests.get(api_url).json()

        for item in tqdm(response, desc=f"Downloading {label} images"):
            img_url = item["urls"]["regular"]

            filename = f"img_{index:04d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)

            try:
                img_data = requests.get(img_url, stream=True)
                if img_data.status_code == 200:
                    with open(filepath, "wb") as f:
                        shutil.copyfileobj(img_data.raw, f)

                    rows.append((filename, label))
                    index += 1

            except Exception as e:
                print("Error downloading:", img_url)

    return rows, index

def main():
    print("[INFO] Starting dataset download...")

    person_rows, idx = download_images(PERSON_URLS, "person", 0)
    no_person_rows, idx = download_images(NO_PERSON_URLS, "no_person", idx)

    # Write labels.csv
    with open(LABELS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(person_rows + no_person_rows)

    print("[INFO] Dataset download complete.")
    print("[INFO] labels.csv has been created.")

if __name__ == "__main__":
    main()
