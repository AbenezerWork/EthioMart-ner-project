# File: src/train_ner.py
#
# Description:
# This script fine-tunes a pre-trained transformer model (e.g., XLM-Roberta)
# for Named Entity Recognition (NER) on a custom Amharic dataset.
# It handles data loading, preprocessing, training, and model saving.
#
# Setup Instructions:
# 1. Make sure you have a Python virtual environment set up and activated.
# 2. Install the required libraries by running the following command in your terminal:
#    pip install torch transformers[torch] datasets seqeval accelerate -U
# 3. Place your labeled data file (e.g., 'first_50.conll') inside the 'data' directory.
# 4. Run the script from the root of your project directory:
#    python src/train_ner.py

import os
import numpy as np
import torch
from datasets import DatasetDict, Dataset, Features, Value, ClassLabel, Sequence
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import seqeval.metrics
from seqeval.scheme import IOB2

def read_custom_conll(file_path):
    """
    Reads a CoNLL-style file with two columns (token and NER tag)
    and returns lists of tokens and tags for each sentence.
    This function correctly handles comments and blank lines.
    """
    tokens_list = []
    ner_tags_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        current_tokens = []
        current_tags = []
        for line in f:
            line = line.strip()
            # A blank line or a comment indicates the end of a sentence.
            if line.startswith("#") or line == "":
                if current_tokens:
                    tokens_list.append(current_tokens)
                    ner_tags_list.append(current_tags)
                    current_tokens = []
                    current_tags = []
            else:
                try:
                    # Assumes space-separated token and tag
                    token, tag = line.split()
                    current_tokens.append(token)
                    current_tags.append(tag)
                except ValueError:
                    # Skips any malformed lines that don't have two columns
                    print(f"Skipping malformed line: {line}")
    
    # Add the last sentence if the file doesn't end with a blank line
    if current_tokens:
        tokens_list.append(current_tokens)
        ner_tags_list.append(current_tags)
        
    return {"tokens": tokens_list, "ner_tags": ner_tags_list}

def main():
    """
    Main function to orchestrate the NER model fine-tuning process.
    """
    # --- 1. Define Constants and Parameters ---
    # You can switch to "xlm-roberta-base" for a larger, potentially more accurate model.
    MODEL_CHECKPOINT = "distilbert-base-multilingual-cased" 
    DATA_FILENAME = "first_50.conll" 
    OUTPUT_DIR = "amharic-ner-model-script-output"

    # Training Hyperparameters
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    WEIGHT_DECAY = 0.01

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device.upper()} ---")
    if device == "cpu":
        print("WARNING: CUDA (GPU) not available. Training will be very slow.")

    # --- 2. Load and Prepare the Dataset ---
    # Construct a robust path to the data file, assuming the script is run from the project root.
    data_file_path = os.path.join("data", DATA_FILENAME)
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found at {data_file_path}. Make sure it's in the 'data' folder.")
    
    print(f"✅ Loading data from: {data_file_path}")
    data_dict = read_custom_conll(data_file_path)

    # --- 3. Create Label Mappings ---
    label_list = sorted(list(set(tag for sen_tags in data_dict["ner_tags"] for tag in sen_tags)))
    class_labels = ClassLabel(names=label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    print(f"\nEntity Labels Found: {label_list}")

    # --- 4. Create Hugging Face Dataset Object ---
    features = Features({
        "tokens": Sequence(Value(dtype='string')),
        "ner_tags": Sequence(class_labels)
    })
    raw_dataset = Dataset.from_dict(data_dict, features=features)
    train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'eval': train_test_split['test']
    })
    print("\nDataset loaded and split successfully:")
    print(dataset)

    # --- 5. Load Tokenizer and Preprocess Data ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("\nTokenizing and aligning labels...")
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    # --- 6. Load Model ---
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    print(f"\nModel '{MODEL_CHECKPOINT}' loaded.")

    # --- 7. Define Training Arguments and Metrics ---
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True if device == "cuda" else False # Enable mixed precision only on GPU
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        f1 = seqeval.metrics.f1_score(true_labels, true_predictions, average="weighted", scheme=IOB2, mode="strict")
        precision = seqeval.metrics.precision_score(true_labels, true_predictions, average="weighted", scheme=IOB2, mode="strict")
        recall = seqeval.metrics.recall_score(true_labels, true_predictions, average="weighted", scheme=IOB2, mode="strict")
        return {"precision": precision, "recall": recall, "f1": f1}

    # --- 8. Initialize Trainer and Start Training ---
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # --- 9. Save the Final Model ---
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✅ Fine-tuned model and tokenizer saved to: {final_model_path}")

if __name__ == "__main__":
    main()
