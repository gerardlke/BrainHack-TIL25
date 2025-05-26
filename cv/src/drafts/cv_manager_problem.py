from typing import Any, List, Dict
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import torch
import ultralytics

import os


class CVManager:

    def __init__(self):
        # Initialize and load the YOLOv8x model
        self.model = YOLO("./yolov8m_til-ai/weights/mcdwkr.pt")

    def cv(self, image: bytes) -> List[Dict[str, Any]]:
        """Performs object detection on an image."""
        # Convert image bytes to NumPy array + get width/height
        img_np, width, height = self._bytes_to_image(image)

        results = self.model(img_np)

        return self._process_predictions(results, width, height)

    def _bytes_to_image(self, image_bytes: bytes) -> tuple[np.ndarray, int, int]:
        """Convert image bytes to a NumPy array and return width/height."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)
        height, width = np_image.shape[:2]
        return np_image, width, height

    def _process_predictions(
        self, results, image_width: int, image_height: int
    ) -> List[Dict[str, Any]]:
        """Process YOLO results into expected output format without rounding bbox values."""

        all_predictions = []

        for result in results:
            image_predictions = []

            if result.boxes is not None:
                for detection in result.boxes.data:
                    x1, y1, x2, y2, conf, class_id = detection.tolist()
                    
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1

                    image_predictions.append({
                        "bbox": [x, y, w, h],
                        "category_id": int(class_id)
                    })

            all_predictions.append(image_predictions)

        # Return a single dictionary with all predictions grouped by image
        return {"predictions": all_predictions}




