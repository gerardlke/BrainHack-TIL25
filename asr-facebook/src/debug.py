import json
import os
import torch
import torchaudio
import numpy as np
from scipy.signal import resample, lfilter, butter
import librosa
from torch.utils.data import DataLoader

from dataclasses import dataclass
from pathlib import Path
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from datasets import (
    load_dataset,
    Dataset,
    Audio
)



# Configuration
MODEL_NAME = '/home/jupyter/mcdonalds-workers/models/asr-finetuned-model/train11'
DATA_DIR = '/home/jupyter/novice/asr'
JSONL_PATH = os.path.join(DATA_DIR, 'asr.jsonl')
SAMPLE_RATE = 16000
OUTPUT_DIR = '/home/jupyter/mcdonalds-workers/models/asr-finetuned-model/train12'
TEST_SIZE = 0.1

# Load processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# Load and prepare 1 file
def load_jsonl(path, max_items=1):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        selected_lines = lines[:max_items]
        data = [json.loads(line) for line in selected_lines]
    return data

data = load_jsonl(JSONL_PATH, max_items=1)

for entry in data:
    entry['path'] = os.path.join(DATA_DIR, entry['audio'])
    entry['transcript'] = entry['transcript'].upper()

dataset = Dataset.from_list(data)
dataset = dataset.cast_column('path', Audio(sampling_rate=SAMPLE_RATE))

# Preprocessing
def preprocess(batch):  
    waveform = batch['path']['array']        # numpy ndarray
    sample_rate = batch['path']['sampling_rate']

    # Mono conversion if needed
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample to 16kHz
    target_sr = 16000
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)

    # Bandpass filter 300-3400 Hz
    lowcut, highcut = 300.0, 3400.0
    fs = target_sr
    order = 5
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    waveform = lfilter(b, a, waveform)

    # Pre-emphasis filter
    waveform = lfilter([1, -0.97], 1, waveform)

    # Trim silence
    waveform, _ = librosa.effects.trim(waveform, top_db=20)

    # Noise gate using RMS energy
    energy = librosa.feature.rms(y=waveform)[0]
    mask = energy > 0.01
    if np.any(mask):
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        waveform = waveform[start*512:end*512]

    # Normalize waveform
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-5)

    # Convert to float32
    waveform = waveform.astype(np.float32)

    # Get input values for model
    batch['input_values'] = processor(waveform, sampling_rate=target_sr).input_values[0]    
        
    # Tokenize transcript for labels
    batch['labels'] = processor.tokenizer(batch['transcript']).input_ids
    
    print(batch['transcript'])
    
    return batch

# Split dataset if needed
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
train_dataset = dataset.select([0])

#split = dataset.train_test_split(test_size=TEST_SIZE)
#train_dataset, eval_dataset = split['train'], split.get('test', None)
    

# Data collator for dynamic padding
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        input_features = [{'input_values': f['input_values']} for f in features]
        label_features = [{'input_ids': f['labels']} for f in features]

        # Pad input audio
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors='pt'
        )

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors='pt'
            )

        # Replace non-attended tokens with -100 for CTC loss
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch['labels'] = labels

        # Optional debug print
        input_lengths = batch['input_values'].shape[1]
        label_lengths = (labels != -100).sum(dim=1)

        # Estimate Wav2Vec2 output length based on conv stride (default ~320x downsampling)
        model_stride = 320  # Wav2Vec2Base: conv stride product = 320
        feature_lengths = input_lengths // model_stride

        if (label_lengths > feature_lengths).any():
            raise ValueError(
                f"[CTC] At least one label is too long!\n"
                f"Input length: {input_lengths} → Features: {feature_lengths}\n"
                f"Label lengths: {label_lengths.tolist()}"
            )
            
        else: print ("yay")

        return batch

    


collator = DataCollatorCTCWithPadding(processor=processor)
sample_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collator)

batch = next(iter(sample_loader))
print(batch['labels'])







