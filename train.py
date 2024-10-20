import argparse
import logging
import os

import torch
from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")

    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset
    train_dataset = load_dataset('csv', data_files=os.path.join(args.train_dir, 'train.csv'))['train']
    eval_dataset = load_dataset('csv', data_files=os.path.join(args.validation_dir, 'validation.csv'))['train']

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=6)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(args.model_dir)

if __name__ == "__main__":
    main()