import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import numpy as np
import io
import soundfile as sf
import torch
import base64
import librosa
import string

class ASRManager:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "whisper-finetuned"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.generation_config.forced_decoder_ids = None
        self.model.to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
    def asr(self, audio_bytes: bytes) -> str:
            # Load audio from bytes using soundfile
            audio_np, sr = sf.read(io.BytesIO(audio_bytes))  # shape: (samples,) or (samples, channels)

            # Convert to mono if stereo
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)

            # Resample to 16000 Hz if needed
            if sr != 16000:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

            # Convert to float32
            audio_np = audio_np.astype(np.float32)

            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )

            transcription = pipe(audio_np)
        
            
            return transcription
