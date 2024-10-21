import torch
from transformers import Trainer, TrainingArguments, logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import load_dataset
from tqdm import tqdm


def train_model(model, tokenizer, model_name, verbose=False, resume_from_checkpoint=None):
    if verbose:
        logging.set_verbosity_info()
    else:
        logging.set_verbosity_warning()
    # Load the dataset
    if verbose:
        print("Loading dataset...")
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
        disable_tqdm=not verbose,  # Enable tqdm if verbose
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Check if resuming from a checkpoint
    if resume_from_checkpoint:
        if verbose:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=f'models/{model_name}_fine_tuned/checkpoint-{resume_from_checkpoint}')
    else:
        trainer.train()

    if verbose:
        print(f"Model fine-tuned and saved at './models/{model_name}_fine_tuned'")

    # Save the model
    trainer.save_model(f'models/{model_name}_fine_tuned')
