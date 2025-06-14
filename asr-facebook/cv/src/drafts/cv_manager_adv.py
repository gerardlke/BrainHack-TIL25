"""Manages the CV model."""

import os, json, time
from typing import Any
from ultralytics import YOLO, RTDETR
from PIL import Image
import numpy as np
import io, base64
import cv2

class CVManager:
    
    #initialise model
    def __init__(self):
        
        self.model = YOLO("./yolov8m_til-ai/weights/mcdwkr.pt")
    
    #yolo to coco
    def yolo2xywh(self, detection):
        cx = detection[0]
        cy = detection[1]
        w = detection[2]
        h = detection[3]
        
        # top left of bbox
        x = cx - (w/2)
        y= cy - (h/2)
        
        return [x, y, w, h]
    
    def cv(self, img: bytes) -> list[dict[str, Any]]:
        
        #convert bytes to PIL img
        #decode bytes to read img
        img_np = np.array(Image.open(io.BytesIO(img)).convert("RGB"))
        img_self = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        #inference
        results = self.model(source=img_self, conf=0.5, save=false, iou=0.5)
        
        #process results
        predictions = []
        for result in results:
            
                for detection in result.boxes:
                    img_prediction = {
                        # 'bbox': self.yolo_to_xywh(box.xywh.cpu().numpy()),
                        'bbox': self.yolo2xywh(box.xywh[0].tolist()),
                        'category_id': int(box.cls.item()), 
                        # 'category_id': int(box.cls[0]),  
                    }
                    predictions.append(img_predictions)

        return predictions
    











