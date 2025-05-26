from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import requests
import jiwer
import time
import base64

import os
import gdown
from typing import List

from asr_manager import ASRManager

# Download a sample WAV file (spoken English, short)
SAMPLE_URLS = [
    "https://cdn.jsdelivr.net/gh/Jakobovski/free-spoken-digit-dataset@master/recordings/1_george_0.wav",
    "https://cdn.jsdelivr.net/gh/Jakobovski/free-spoken-digit-dataset@master/recordings/7_lucas_0.wav"
]

# Download WAV files from personal google drive
gdrive_urls = ["https://drive.google.com/uc?id=1x_wlbIG8vj-EdZ--wES7T1txl9S73rqM", "https://drive.google.com/uc?id=1s9nMZS89KuPwGG0AyLFsVR9b6xl34r_u"]

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

def download_audio(urls: list[str]) -> list[bytes]:
    res = []
    for idx, url in enumerate(urls):
        print(f"Downloading audio from {url}")
        response = requests.get(url)
        response.raise_for_status()
        audio_bytes = response.content
        audio_b64 = base64.b64encode(audio_bytes)
        res.append(audio_b64)

    return res

def score_asr(ground_truth: str, prediction: str) -> float:
    # Match the competition scoring logic
    wer_transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": " "}),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    return 1 - jiwer.wer(
        [ground_truth],
        [prediction],
        truth_transform=wer_transforms,
        hypothesis_transform=wer_transforms,
    )

def main():
    asr_model = ASRManager()

    # Step 1: Download sample audio
    #audio_bytes = download_audio_gdrive(gdrive_urls) # use gdrive audio
    audio_bytes = download_audio(SAMPLE_URLS)

    for idx, audio in enumerate(audio_bytes):
        # Step 2: Set expected transcription
        ground_truth = "one"

        # Step 3: Load and run ASR model
        start = time.time()
        prediction = asr_model.asr(audio)
        end = time.time()

        # Step 4: Score
        print("\n=== Transcription ===")
        print("Predicted:", prediction)
        print("Ground Truth:", ground_truth)

#        accuracy = score_asr(ground_truth, prediction)
#        print("\n=== Metric ===")
#        print("1 - WER:", accuracy)
#        print("Elapsed time:", round(end - start, 4))

if __name__ == "__main__":
    main()

