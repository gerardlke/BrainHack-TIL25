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
        self.model = YOLO("yolov11x_1024_freeze12/weights/best.pt")

    def cv(self, image: bytes) -> List[Dict[str, Any]]:
        """Performs object detection on an image."""
        # Convert image bytes to NumPy array + get width/height
        img_np, width, height = self._bytes_to_image(image)

        results = self.model(img_np, conf=0.5, save=False, iou=0.5)

        return self._process_predictions(results, width, height)

    def _bytes_to_image(self, image_bytes: bytes) -> tuple[np.ndarray, int, int]:
        """Convert image bytes to a NumPy array and return width/height."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)
        height, width = np_image.shape[:2]
        return np_image, width, height
    
    def yolo2xywh(self, detection):
        cx = detection[0]
        cy = detection[1]
        w = detection[2]
        h = detection[3]
        
        x = cx - (w/2)
        y = cy - (h/2)
        
        return [x, y, w, h]
    
    def _process_predictions(
        self, results, image_width: int, image_height: int
    ) -> List[Dict[str, Any]]:
        """Process YOLO results into expected output format without rounding bbox values."""
        image_predictions = []

        for result in results:
            
            if result.boxes is not None:            
                for detection in result.boxes:
                    image_predictions.append({
                        "bbox": self.yolo2xywh(detection.xywh[0].tolist()),
                        "category_id": int(detection.cls.item())
                    })

        # Return a single dictionary with all predictions grouped by image
        return image_predictions




