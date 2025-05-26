import json
import os
import torch
import torchaudio
import numpy as np
from scipy.signal import resample, lfilter, butter
import librosa

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

from torch.utils.data import DataLoader



# Configuration
MODEL_NAME = 'facebook/wav2vec2-base-960h'
DATA_DIR = '/home/jupyter/novice/asr'
JSONL_PATH = os.path.join(DATA_DIR, 'asr.jsonl')
SAMPLE_RATE = 16000
OUTPUT_DIR = '/home/jupyter/mcdonalds-workers/models/asr-finetuned-model/train100'
TEST_SIZE = 0.1

# Load processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# Load and prepare full dataset
def load_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    return data

# Load and prepare top half of the dataset
#def load_jsonl(path, take_fraction=0.04):
#    with open(path, 'r') as f:
#        lines = [line.strip() for line in f if line.strip()]
#        num_to_take = int(len(lines) * take_fraction)
#        selected_lines = lines[:num_to_take]
#        data = [json.loads(line) for line in selected_lines]
#    return data

data = load_jsonl(JSONL_PATH)



for entry in data:
    entry['path'] = os.path.join(DATA_DIR, entry['audio'])
    entry['transcript'] = entry['transcript'].lower()

dataset = Dataset.from_list(data)
dataset = dataset.cast_column('path', Audio(sampling_rate=SAMPLE_RATE))


# Preprocessing
def preprocess(batch):
    audio = batch['path']
    batch['input_values'] = processor(audio['array'], sampling_rate=SAMPLE_RATE).input_values[0]
    
    # Use the tokenizer directly for labels
    batch['labels'] = processor.tokenizer(batch['transcript'].upper()).input_ids    
    return batch

# Split dataset if needed
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
split = dataset.train_test_split(test_size=TEST_SIZE)
train_dataset, eval_dataset = split['train'], split.get('test', None)

# Data collator for dynamic padding
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        input_features = [{'input_values': f['input_values']} for f in features]
        label_features = [{'input_ids': f['labels']} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors='pt'
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors='pt'
            )

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch['labels'] = labels

        return batch

# Load model
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    ctc_loss_reduction='mean',
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True,
    per_device_train_batch_size=2,
    eval_strategy='epoch',
    num_train_epochs=80,
    gradient_accumulation_steps=2,
    fp16=True,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    logging_first_step=True,
    learning_rate=1e-4,
    warmup_steps=0,
    disable_tqdm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=DataCollatorCTCWithPadding(processor=processor),
)

print('Starting Training.')

trainer.train()

print('Training Completed.')

# Save model & processor
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print('Saved model checkpoints.')