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
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import io
import base64
import torchaudio
import numpy as np
import soundfile as sf
from scipy.signal import resample
        
class ASRManager:

    def __init__(self, encoder_model="facebook/wav2vec2-base-960h", decoder_type="ctc"):
        # This is where you can initialize your model and any static
        # configurations.
        self.device = 0 if torch.cuda.is_available() else -1
        # self.pipeline = pipeline(
        #     "automatic-speech-recognition",
        #     model="openai/whisper-small",
        #     device=self.device,
        #     generate_kwargs={"language": "english"}  # Force English transcription
        # )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Wav2Vec2Processor.from_pretrained(encoder_model)

        if decoder_type == "ctc":
            self.model = Wav2Vec2ForCTC.from_pretrained(encoder_model).to(self.device)
        elif decoder_type == "seq2seq":
            # Placeholder: You can load a custom attention-based model here
            raise NotImplementedError("Sequence-to-sequence decoder is not implemented in this template.")
        else:
            raise ValueError("Unsupported decoder type: choose 'ctc' or 'seq2seq'")

        self.decoder_type = decoder_type


#     def preprocess_audio(self, audio_bytes: bytes):
#         """Converts bytes to a waveform and resamples to 16kHz mono."""
#         # try:
#         with torch.no_grad():
#             # Decode b64 audio into bytes and creates a buffer in memory to store it (torch audio only works with filepaths or file like objects)
#             audio_bytes = io.BytesIO(audio_bytes)

#             # Converts raw audio bytes into torch tensors and retrieves sample rate (frequency)
#             waveform, sample_rate = torchaudio.load(audio_bytes, format="wav")

#             # Converts stereo or multi-channel audio to mono by averaging channels (but maybe point of experimentation)
#             if waveform.shape[0] > 1:
#                 waveform = waveform.mean(dim=0, keepdim=True)

#             # Resample to 16kHz if needed
#             if sample_rate != 16000:
#                 resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#                 waveform = resampler(waveform)

#             # Normalize to [-1.0, 1.0]
#             max_val = waveform.abs().max()
#             if max_val > 0:
#                 waveform = waveform / max_val

#             # Remove extra dimensions and return numpy array
#             waveform = waveform.squeeze().numpy()

#         return waveform
#         # except Exception as e:
#         #     raise Exception(f'Error preprocessing audio: {e}')
        
        
    def preprocess_audio(self, audio_bytes: str):  
        """Decodes base64-encoded WAV, converts to mono, resamples to 16kHz, and returns numpy array."""
        # Wrap bytes in a buffer
        audio_buffer = io.BytesIO(audio_bytes)

        # Read using soundfile
        waveform, sample_rate = sf.read(audio_buffer)

        # If stereo/multi-channel, average to mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample to 16kHz if necessary
        target_sr = 16000
        if sample_rate != target_sr:
            num_samples = int(len(waveform) * target_sr / sample_rate)
            waveform = resample(waveform, num_samples)

        # Normalize to [-1, 1]
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-5)

        return waveform.astype(np.float32)

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """
        audio_bytes = self.preprocess_audio(audio_bytes)

        # Method 1
        # result = self.pipeline(audio_bytes)
        # print('result', result)
        # return result["text"]

        # Method 2
        inputs = self.processor(audio_bytes, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        print("Input values shape:", input_values.shape)  # Expect: (1, ~16000*T)
        print("Logits shape:", logits.shape)              # Expect: (1, T, vocab_size)
        print("Top logits sample:", logits[0, :5].argmax(dim=-1))  # First 5 predictions

        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.strip()
