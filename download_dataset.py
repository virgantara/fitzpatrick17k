import os
import csv
import requests
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
CSV_FILE = "fitzpatrick17k.csv"  # replace with your CSV filename
OUTPUT_DIR = "dataset"     # base directory for downloaded images

# === Create base directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Function to download image ===
def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed: {url} (status {response.status_code})")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# === Read CSV and process ===
with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in tqdm(reader, desc="Downloading images"):
        label = row["label"].strip().replace(" ", "_").lower() or "unknown"
        url = row["url"].strip()
        filename = row["url_alphanum"].strip()

        # Create label directory if not exists
        label_dir = Path(OUTPUT_DIR) / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Define save path
        save_path = label_dir / filename

        # Skip if already exists
        if save_path.exists():
            continue

        # Download image
        download_image(url, save_path)

print(" Download complete.")
