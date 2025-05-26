import torch
from doctr.models import ocr_predictor, crnn_vgg16_bn, db_resnet50
from doctr.io import DocumentFile
from doctr.datasets import VOCABS
import time
class OCRManager:
    def __init__(self,
                 reco_model_path: str = "./text_recognition.pt",
                 det_model_path: str = "./text_detection.pt",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # Device 
        self.device = device

        # Initialize text detection model
        det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
        det_params = torch.load('./text_detection.pt', map_location=device)
        det_model.load_state_dict(det_params)


        # Initialize the text recognition model
        reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False,vocab=VOCABS["english"])
        reco_params = torch.load('./text_recognition.pt', map_location=device)
        reco_model.load_state_dict(reco_params)
        
        det_model.eval()
        reco_model.eval()
        
        det_model.to(self.device)
        reco_model.to(self.device)



        # Initialize OCR model
        self.predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model, pretrained=False).cuda().half()


    def ocr(self, image_bytes: bytes) -> str:
        """Performs OCR on an image of a document.
        
        Args:
            image: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """

        # Load image from bytes
        doc = DocumentFile.from_images(image_bytes)

        # Makes the predictions 
        with torch.no_grad():
            result = self.predictor(doc)
            
        # Convert prediction output to string
        text = result.render()
        
        return text
