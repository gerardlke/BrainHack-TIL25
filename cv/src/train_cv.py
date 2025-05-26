'''train_yolov.py'''
from ultralytics import YOLO

# def train(epochs):
#     model = YOLO("yolo11x.pt")

#     model.train(
#         data="/home/jupyter/mcdonalds-workers/BrainHack-TIL25/cv/src/cv_data.yaml",
#         epochs=x,                    # Number of training epochs
#         imgsz=800,                   # Image resize
#         batch=8,                      # Batch size
#         workers=4,                    # Number of CPU workers for data loading
#         device=0,                     # GPU to use
# #       precision=16,                 # Mixed precision training
#         project="/home/jupyter/mcdonalds-workers/BrainHack-TIL25/cv/src",   # Save directory
#         name="yolo11x_til-ai",        # Model name
#         verbose=True
#     )


def train(epochs):
    model = YOLO("yolo11x.pt")

    model.train(
        data="/home/jupyter/mcdonalds-workers/BrainHack-TIL25/cv/src/cv_augmented.yaml",
        epochs=epochs,
        imgsz=1024,
        batch=8,
        workers=4,
        device=0,
        freeze=12, # 12 layers in backbone for yolov11l/x
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=3,
        patience=10,
        weight_decay=0.0005,
        label_smoothing=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        perspective=0.0005, flipud=0.1, fliplr=0.5,
        mosaic=1.0, mixup=0.2,
        amp=True,
        save=True,
        project="/home/jupyter/mcdonalds-workers/BrainHack-TIL25/cv/src",
        name="yolov11x_1024_freeze12_augmented",
        verbose=True
    )

if __name__ == "__main__":
    train(epochs=30)
