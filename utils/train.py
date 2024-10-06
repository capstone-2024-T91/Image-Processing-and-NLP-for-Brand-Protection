import torch
from transformers import Trainer, TrainingArguments
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import load_dataset


def train_model(model, tokenizer, model_name, verbose=False):
    # Load the dataset
    train_dataset, eval_dataset = load_dataset(tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'models/{model_name}_fine_tuned',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='steps',
        save_steps=500,
        eval_steps=500,
        logging_dir='./logs',
        logging_steps=100,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

    if verbose:
        print(f"Model fine-tuned and saved at './models/{model_name}_fine_tuned'")

    # Save the model
    trainer.save_model(f'models/{model_name}_fine_tuned')
