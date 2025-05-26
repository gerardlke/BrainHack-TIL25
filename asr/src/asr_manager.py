"""WHISPER"""

import whisper
from whisper.model import Whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import io
import soundfile as sf
import torch
import base64
import librosa
import string
import re

class ASRManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperForConditionalGeneration.from_pretrained("./whisper2").to(self.device)
        self.processor = WhisperProcessor.from_pretrained("./whisper2")
        
        self.model.generation_config.language = "en"
        self.model.generation_config.task = "transcribe"
        
        self.model.generation_config.forced_decoder_ids = None
       
        self.model.eval()
        
        
    def clean_transcription(self, text):
        # Remove non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        # Normalize punctuation spacing
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        deduped = []
        for i, sent in enumerate(sentences):
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            sent_clean = re.sub(r"([.!?])\1+", r"\1", sent_clean)  # Collapse repeated punctuation
            # Deduplicate with case-insensitive substring check
            if not any(sent_clean.lower() in s.lower() or s.lower() in sent_clean.lower() for s in deduped):
                deduped.append(sent_clean)

        return " ".join(deduped).strip()
        
    
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
        
        inputs = self.processor(audio_np, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        input_features = inputs.input_features.to(self.device)
    
        with torch.no_grad():           
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=63,
                num_beams=1,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
            
        
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        cleaned_transcription = self.clean_transcription(transcription)

        
        return cleaned_transcription