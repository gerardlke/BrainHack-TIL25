import requests
import time
import base64

import os
import gdown
from typing import List
import json

from asr_manager import ASRManager


# Download WAV files from personal google drive
gdrive_urls = ["https://drive.google.com/uc?id=1WSSQ7paBxlSz7dEhfco1CSg5BFSyAhJQ"]
#, "https://drive.google.com/uc?id=1s9nMZS89KuPwGG0AyLFsVR9b6xl34r_u"]

# Function to pull files from google drive kdk
def download_audio_gdrive(urls: List[str]) -> List[bytes]:
    res = []
    for idx, url in enumerate(urls):
        print(f"Downloading audio from {url}")
        
        # Extract file ID from the URL
        if "drive.google.com" in url:
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1]
            else:
                raise ValueError(f"Could not extract file ID from URL: {url}")
        else:
            raise ValueError("Only Google Drive URLs are supported.")

        # Download the file using gdown
        output_file = f"audio_{idx}.wav"
        gdown.download(id=file_id, output=output_file, quiet=False)

        # Read file as bytes
        with open(output_file, "rb") as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes)
            res.append(audio_b64)

        # Optionally remove the temp file
        os.remove(output_file)

    return res



def main():
    asr_model = ASRManager()

    # Step 1: Download sample audio
    audio_bytes = download_audio_gdrive(gdrive_urls) # use gdrive audio
    
    predictions=[]

    for idx, audio in enumerate(audio_bytes):
        # Step 2: Set expected transcription
        ground_truth = "one"

        # Step 3: Load and run ASR model
        start = time.time()
        audio = base64.b64decode(audio)
        prediction = asr_model.asr(audio)
        end = time.time()
        
        predictions.append(prediction)
    
    hey={"predictions": predictions}

    print("\n=== Transcription ===")
    print("outputdict", hey)
    print("Ground Truth:", ground_truth)
        
        

#        accuracy = score_asr(ground_truth, prediction)
#        print("\n=== Metric ===")
#        print("1 - WER:", accuracy)
#        print("Elapsed time:", round(end - start, 4))

if __name__ == "__main__":
    main()
