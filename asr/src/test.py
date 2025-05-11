import requests
import jiwer
import time
import base64

from asr_manager import ASRManager

# Download a sample WAV file (spoken English, short)
SAMPLE_URLS = [
    "https://cdn.jsdelivr.net/gh/Jakobovski/free-spoken-digit-dataset@master/recordings/0_george_0.wav",
    "https://cdn.jsdelivr.net/gh/Jakobovski/free-spoken-digit-dataset@master/recordings/1_george_1.wav",
]

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
    audio_bytes = download_audio(SAMPLE_URLS)

    for idx, audio in enumerate(audio_bytes):
        # Step 2: Set expected transcription
        ground_truth = "zero"

        # Step 3: Load and run ASR model
        start = time.time()
        prediction = asr_model.asr(audio)
        end = time.time()

        # Step 4: Score
        print("\n=== Transcription ===")
        print("Predicted:", prediction)
        print("Ground Truth:", ground_truth)

        accuracy = score_asr(ground_truth, prediction)
        print("\n=== Metric ===")
        print("1 - WER:", accuracy)
        print("Elapsed time:", round(end - start, 4))

if __name__ == "__main__":
    main()

