"""
Manages the ASR model.

Version 1: Using Whisper 
    - End to end speech to text ASR model
    - Transformer-based encoder decoder 
        - Encoder takes audio spectrograms (using mel spectrogram) as input
        - Decoder uses masked self attention and cross attention to generate outputs autoregressively
        - Final linear layer with softmax activation to predict probability distribution over vocab of text tokens 
    - Note that whisper takes in 
"""

import torch
import time
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import io
import base64
import torchaudio
import numpy as np
import soundfile as sf
from scipy.signal import resample, lfilter, butter
import librosa
#from pyctcdecode import BeamSearchDecoderCTC

import random
import matplotlib.pyplot as plt


class ASRManager:

    def __init__(self, encoder_model="facebook/wav2vec2-base-960h", decoder_type="ctc", lm_path=None): # '/home/jupyter/mcdonalds-workers/models/4-gram.arpa'
        """To initialize model and any static configurations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(encoder_model)
        self.decoder_type = decoder_type

        # Load acoustic model
        if decoder_type == "ctc":
            self.model = Wav2Vec2ForCTC.from_pretrained(encoder_model).to(self.device)
        elif decoder_type == "seq2seq":
            # Placeholder: You can load a custom attention-based model here
            raise NotImplementedError("Sequence-to-sequence decoder is not implemented in this template.")
        else:
            raise ValueError("Unsupported decoder type: choose 'ctc' or 'seq2seq'")

        # Initialize language model decoder (optional)
        self.lm_decoder = None
        if lm_path is not None:
            vocab_list = list(self.processor.tokenizer.get_vocab().keys())
            vocab_list = [token.lower() if token not in ["<pad>", "<s>", "</s>"] else token for token in vocab_list]
            self.lm_decoder = BeamSearchDecoderCTC(vocab_list, kenlm_model_path=lm_path)

    def plot_waveform(self, waveform, name:str = 'Filename', title:str = "Waveform"):
        # If stereo, convert to mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Create time axis in seconds
        time = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))

        # Plot
        plt.figure(figsize=(14, 4))
        plt.plot(time, waveform, linewidth=0.8)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./visualisations/{name}.png')
    
    def preprocess_audio(self, audio_bytes: str):  
        """Decodes base64-encoded WAV, converts to mono, resamples to 16kHz, and returns numpy array."""
        # Load audio
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_buffer)

        # Mono conversion
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample to 16kH
        target_sr = 16000
        if sample_rate != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)

        # Normalize to [-1, 1]
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-5)
        
        return waveform.astype(np.float32)
    

    def preprocess_audio2(self, audio_bytes: str):  
        """Decodes base64 WAV, converts to mono, resamples to 16kHz, denoises, and normalizes."""
        # Load audio
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_buffer)

        # Mono conversion
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample to 16kHz
        target_sr = 16000
        if sample_rate != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
            
        # Apply band pass filter
        lowcut=300.0
        highcut=3400.0
        fs=16000
        order=5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        waveform = lfilter(b, a, waveform)

        # Apply pre-emphasis filter (boosts high-frequency clarity)
        waveform = lfilter([1, -0.97], 1, waveform)

        # Trim leading/trailing silence (helps reduce padding during CTC decoding)
        waveform, _ = librosa.effects.trim(waveform, top_db=20)

        # Simple noise gate: zero out very low energy regions
        energy = librosa.feature.rms(y=waveform)[0]
        mask = energy > 0.01  # adjust this based on ambient noise level
        if np.any(mask):
            start = np.argmax(mask)
            end = len(mask) - np.argmax(mask[::-1])
            waveform = waveform[start*512:end*512]  # librosa RMS uses hop_length=512 by default

        # Normalize waveform
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-5)

        return waveform.astype(np.float32)

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """
        # Preprocess audio and convert into a torch tensor
        num = random.randint(1, 10000)

        audio_bytes = base64.b64decode(audio_bytes) # decode base64 WAV
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_buffer)
        # self.plot_waveform(waveform, name=f'{num}_original', title="Raw Input Audio")

        a = time.time()
        waveform = self.preprocess_audio(audio_bytes)
        # self.plot_waveform(waveform1, name=f'{num}_processed_11', title="Preprocessed audio 1")
        b = time.time()
        
        # TEST 1
        waveform_tensor = torch.from_numpy(waveform)

        # Converts raw audio into tokenized inputs (including feature extraction and padding if needed)
        inputs = self.processor(
            waveform_tensor, sampling_rate=16000, return_tensors="pt"
        )

        # Moving inputs to device
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Run encoder to get logits
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits

        # Decoding logits into output (current is either CTC or LM)
        predicted_ids = torch.argmax(logits, dim=-1)
        if self.lm_decoder is not None:
            transcription = self.lm_decoder.decode(logits.cpu().numpy()[0])
        else:
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        #print('\n==Predictions==')
        #print('Prediction:', transcription.strip())
        
 


