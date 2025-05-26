"""Manages the CV model."""

import os, json, time
from typing import Any
from ultralytics import YOLO, RTDETR
from PIL import Image
import numpy as np
import io, base64
import cv2

class CVManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.

        # load model YOLO
        model_path = r'/workspace/best_1024f10mu025_lr0010005_DL.pt'
        self.model = YOLO(model_path)
        
        # load model RTDETR
        # model_path = r'/workspace/rtdetr_quantized_model.pt'
        # self.model = RTDETR(model_path)
        
    def yolo_to_xywh(self, bboxes_yolo):
        x_center = bboxes_yolo[0]  # x_center 
        y_center = bboxes_yolo[1]  # y_center
        width = bboxes_yolo[2]     # width
        height = bboxes_yolo[3]    # height

        # covnert xleft and yleft
        x_left = x_center - (width / 2)  # x_left = cx - w/2
        y_top = y_center - (height / 2)  # y_top = cy - h/2

        return [x_left, y_top, width, height]

        # Convert to (x_left, y_top, width, height)
#         bboxes_yolo[:, 0] -= bboxes_yolo[:, 2] / 2  # x_left = cx - w/2
#         bboxes_yolo[:, 1] -= bboxes_yolo[:, 3] / 2  # y_top = cy - h/2

#         # Result is now in x_left, y_top, w, h format
#         bbox = bboxes_yolo.squeeze().tolist()  # [x_left, y_top, w, h]
    
#         return bbox

    # 
        
#     def should_denoise(self, image, blur_ksize=(7, 7), threshold=12.0, rel_threshold=0.3, check_size=(256, 256)):
#         resized = cv2.resize(image, check_size, interpolation=cv2.INTER_AREA)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         mean = cv2.blur(gray, blur_ksize)
#         diff = gray - mean
#         noise_level = np.std(diff)
        
#         rel_noise = noise_level / (np.std(gray) + 1e-6)
        
#         # noise_level = self.estimate_noise_per_channel(image)
#         # return noise_level > threshold
#         return noise_level > threshold and rel_noise > rel_threshold
    def should_denoise(self, image, threshold=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
        

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
            [{
                "bbox": [x, y, w, h],
                "category_id": category_id
            },...]
        """

        # Your inference code goes here.
        # Convert bytes to PIL Image

        # decode bytes to read image
        image_np = np.array(Image.open(io.BytesIO(image)).convert("RGB"))
        image_itself = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        print(image_itself)
        
        #!! try denoising before predicting:
        # start = time.time()
        # if self.should_denoise(image_itself):
        #     print('denoise here')
        #     image_itself = cv2.fastNlMeansDenoisingColored(image_itself, None, 10, 10, 7, 21)
        # end = time.time()
        # print("time denoising",(end-start) * 10**3, "ms")
        
        # Perform inference
        results = self.model(source=image_itself, conf=0.5, save=False, iou=0.5)

        # Process results
        predictions = []
        for result in results:
            for box in result.boxes:
                bbox = {
                    # 'bbox': self.yolo_to_xywh(box.xywh.cpu().numpy()),
                    'bbox': self.yolo_to_xywh(box.xywh[0].tolist()),
                    'category_id': int(box.cls.item()), 
                    # 'category_id': int(box.cls[0]),  
                }
                predictions.append(bbox)

        return predictions
