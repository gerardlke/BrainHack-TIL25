import os
import cv2
import numpy as np
from pathlib import Path, PurePath
from tqdm import tqdm

IMAGE_FOLDER = "/home/jupyter/BrainHack-TIL25/cvdataset/images/val"
LABEL_FOLDER = "/home/jupyter/BrainHack-TIL25/cvdataset/labels/val"
OUTPUT_IMAGE_FOLDER = "/home/jupyter/BrainHack-TIL25/cvaugmented/images/val"
OUTPUT_LABEL_FOLDER = "/home/jupyter/BrainHack-TIL25/cvaugmented/labels/val"

Path(OUTPUT_IMAGE_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_LABEL_FOLDER).mkdir(exist_ok=True)

def flip_image_and_boxes(image, boxes):
    """Horizontally flip image and bounding boxes."""
    flipped_image = cv2.flip(image, 1)
    flipped_boxes = []
    for box in boxes:
        cls, x_center, y_center, width, height = box
        x_center = round(1.0 - x_center, 6)  # Flip X center
        flipped_boxes.append([cls, x_center, y_center, width, height])
    return flipped_image, flipped_boxes

def load_yolo_labels(label_path):
    """Load YOLO labels from file."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([int(parts[0])] + list(map(float, parts[1:])))
    return boxes

def save_yolo_labels(label_path, boxes):
    """Save YOLO labels to file."""
    with open(label_path, 'w') as f:
        for box in boxes:
            cls, x_center, y_center, width, height = box
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_dataset(image_dir, label_dir):
    image_paths = Path(image_dir).rglob('*.jpg')
    for img_path in tqdm(image_paths, desc='Augmenting dataset...'):
        filename = PurePath(img_path).stem
        label_path = Path(label_dir) / f'{filename}.txt'
        if not os.path.exists(label_path):
            print('None')
            continue

        # Load image and labels
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        boxes = load_yolo_labels(str(label_path))

        # Flip image and boxes
        flipped_img, flipped_boxes = flip_image_and_boxes(image, boxes)

        # Save flipped image and label
        out_img_path = Path(OUTPUT_IMAGE_FOLDER) / f'{filename}_flip.jpg'
        cv2.imwrite(str(out_img_path), flipped_img)

        out_label_path = Path(OUTPUT_LABEL_FOLDER) / f'{filename}_flip.txt'
        save_yolo_labels(str(out_label_path), flipped_boxes)

        # Optionally copy original image/labels too
        out_img_path = Path(OUTPUT_IMAGE_FOLDER) / f'{filename}.jpg'
        cv2.imwrite(str(out_img_path), image)

        out_label_path = Path(OUTPUT_LABEL_FOLDER) / f'{filename}.txt'
        save_yolo_labels(str(out_label_path), boxes)

    print("Augmentation complete.")

# === RUN SCRIPT ===
if __name__ == "__main__":
    augment_dataset(IMAGE_FOLDER, LABEL_FOLDER)