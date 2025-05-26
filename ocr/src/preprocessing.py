import os
import glob
import random
import json
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm

# PATHS TO RESPECTIVE FILES 
data_dir = "/home/jupyter/novice/ocr"
output_dir = "/home/jupyter/ocr_dataset"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
# If directories do not exist
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)

# Get all hocr files and limit to first N cropped words
hocr_files = sorted(glob.glob(os.path.join(data_dir, "*.hocr")))
crop_limit = 9000
all_crops = []

def extract_words_from_hocr(hocr_path):
    with open(hocr_path, "r", encoding="utf8") as f:
        soup = BeautifulSoup(f, "html.parser")
    words = []
    for span in soup.find_all("span", class_="ocrx_word"):
        if "title" not in span.attrs or not span.text.strip():
            continue
        title = span["title"]
        if "bbox" not in title:
            continue
        bbox_str = title.split(";")[0].replace("bbox", "").strip()
        x1, y1, x2, y2 = map(int, bbox_str.split())
        text = span.text.strip()
        words.append(((x1, y1, x2, y2), text))
    return words

crop_id = 0

for hocr_file in tqdm(hocr_files, desc="Processing HOCR"):
    img_file = hocr_file.replace(".hocr", ".jpg")
    if not os.path.exists(img_file):
        continue

    try:
        img = Image.open(img_file).convert("RGB")
        words = extract_words_from_hocr(hocr_file)
    except Exception as e:
        print(f"Failed to process {img_file}: {e}")
        continue

    for (x1, y1, x2, y2), text in words:
        crop = img.crop((x1, y1, x2, y2))
        all_crops.append((crop.copy(), text))  # copy to avoid image closing
        crop_id += 1
        if crop_id >= crop_limit:
            break

    if crop_id >= crop_limit:
        break

print(f"Collected {len(all_crops)} word crops.")

# Shuffle and split
random.shuffle(all_crops)
split_index = int(0.8 * len(all_crops))
train_crops = all_crops[:split_index]
val_crops = all_crops[split_index:]

def save_crops(crops, out_dir):
    labels = {}
    for i, (img, text) in enumerate(crops):
        img_name = f"crop_{i:05d}.jpg"
        img_path = os.path.join(out_dir, "images", img_name)
        img.save(img_path)
        labels[img_name] = text
    # Save labels.json
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

# Save train and val
save_crops(train_crops, train_dir)
save_crops(val_crops, val_dir)

print("Preprocessing Completed. Dataset has been created.")
