import os
import json
from tqdm import tqdm
import shutil

# Paths
COCO_JSON = "/home/jupyter/novice/cv/annotations.json"
IMAGES_DIR = "/home/jupyter/novice/cv/images"
OUTPUT_DIR = "/home/jupyter/BrainHack-TIL25/cvdataset"

# Output folders
train_img_dir = os.path.join(OUTPUT_DIR, "images/train")
val_img_dir = os.path.join(OUTPUT_DIR, "images/val")
train_lbl_dir = os.path.join(OUTPUT_DIR, "labels/train")
val_lbl_dir = os.path.join(OUTPUT_DIR, "labels/val")

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Load COCO JSON
with open(COCO_JSON, "r") as f:
    data = json.load(f)

# Map image ID to metadata
image_info = {img["id"]: img for img in data["images"]}

# Sort image_ids by filename for consistency, then select last 500 for val
sorted_images = sorted(data["images"], key=lambda x: x["file_name"])
val_image_ids = set(img["id"] for img in sorted_images[-500:])

# Process annotations
for ann in tqdm(data["annotations"], desc="Converting annotations"):
    image_id = ann["image_id"]
    bbox = ann["bbox"]
    category_id = ann["category_id"]

    img = image_info[image_id]
    img_name = img["file_name"]
    width, height = img["width"], img["height"]
    stem = os.path.splitext(img_name)[0]

    # Decide split
    if image_id in val_image_ids:
        img_dir = val_img_dir
        lbl_dir = val_lbl_dir
    else:
        img_dir = train_img_dir
        lbl_dir = train_lbl_dir

    # Copy image
    src_img = os.path.join(IMAGES_DIR, img_name)
    dst_img = os.path.join(img_dir, img_name)
    if not os.path.exists(dst_img):
        shutil.copy(src_img, dst_img)

    # Convert to YOLO format
    x, y, w, h = bbox
    xc = (x + w / 2) / width
    yc = (y + h / 2) / height
    nw = w / width
    nh = h / height

    # Write to label file
    label_path = os.path.join(lbl_dir, f"{stem}.txt")
    with open(label_path, "a") as f:
        f.write(f"{category_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
