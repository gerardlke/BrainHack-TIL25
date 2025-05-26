import os
import whisper
from whisper.model import Whisper
import json
import numpy as np
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, List, Dict, Union
import torch
import evaluate


DATA_DIR = '/home/jupyter/novice/asr'
JSONL_PATH = os.path.join(DATA_DIR, 'asr.jsonl')
SAMPLE_RATE = 16000
TEST_SIZE = 0.1

# Load jsonl data into a list of dicts
def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]
    
# Load and prepare small dataset
#def load_jsonl(path, take_fraction=0.01):
#    with open(path, 'r') as f:
#        lines = [line.strip() for line in f if line.strip()]
#        num_to_take = int(len(lines) * take_fraction)
#        selected_lines = lines[:num_to_take]
#        data = [json.loads(line) for line in selected_lines]
#    return data

raw_data = load_jsonl(JSONL_PATH)

# Fix paths and lowercase transcript
for entry in raw_data:
    entry['path'] = os.path.join(DATA_DIR, entry['audio'])
    entry['transcript'] = entry['transcript'].lower()

dataset = Dataset.from_list(raw_data)
dataset = dataset.cast_column("path", Audio(sampling_rate=SAMPLE_RATE))

# Load base model & processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Force language to English and task to transcription
language = "en"
task = "transcribe"
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
model.config.forced_decoder_ids = forced_decoder_ids

# 4. Preprocessing
def preprocess(batch):
    audio = batch['path']
    signal = audio['array']

    # Extract features (input_values) and attention_mask
    inputs = processor.feature_extractor(
        signal,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=True,
        return_tensors="np"
    )
    batch['input_features'] = inputs["input_features"][0]
    batch['attention_mask'] = inputs["attention_mask"][0]

    # Tokenize transcript (labels)
    batch['labels'] = processor.tokenizer(batch['transcript']).input_ids
    return batch

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Split and format
split = dataset.train_test_split(test_size=TEST_SIZE)
train_dataset = split["train"]
eval_dataset = split["test"]

train_dataset.set_format(type="torch", columns=["input_features", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_features", "attention_mask", "labels"])

# Data collator with EOS token
@dataclass
class WhisperDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[torch.Tensor, List[int]]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = []

        for f in features:
            ids = f["labels"]
            ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
            ids += [self.processor.tokenizer.eos_token_id]
            label_features.append({"input_ids": ids})

        # Pad input features and return attention_mask
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_attention_mask=True,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels
        return batch

# Metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    labels = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper3",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    num_train_epochs=5
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=WhisperDataCollatorWithPadding(processor),
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model and processor
model.save_pretrained("./whisper3")
processor.save_pretrained("./whisper3")
