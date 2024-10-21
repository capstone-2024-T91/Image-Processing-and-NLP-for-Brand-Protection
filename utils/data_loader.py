import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sklearn.model_selection import train_test_split
import torch
from tqdm.auto import tqdm


def load_dataset(tokenizer):
    phishing_emails = pd.read_csv("https://media.githubusercontent.com/media/capstone-2024-T91/Image-Processing-and-NLP-for-Brand-Protection/refs/heads/prototype_v3/data/processed/phishing_train_1.csv")
    phishing_emails_2 = pd.read_csv("https://media.githubusercontent.com/media/capstone-2024-T91/Image-Processing-and-NLP-for-Brand-Protection/refs/heads/prototype_v3/data/processed/phishing_train_2.csv")

    # Combine datasets
    emails = pd.concat([phishing_emails, phishing_emails_2], ignore_index=True)
    labels = emails['label'].tolist()
    texts = emails['body'].tolist()

    # Tokenize texts
    # Tokenize texts with progress bar
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors='pt',
    )

    # Convert to torch Dataset
    class EmailDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    # Split dataset
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    # Tokenize with progress bar
    if tokenizer.verbose:
        print("Tokenizing training data...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        return_tensors='pt',
    )
    if tokenizer.verbose:
        print("Tokenizing evaluation data...")
    eval_encodings = tokenizer(
        eval_texts,
        truncation=True,
        padding=True,
        return_tensors='pt',
    )

    train_dataset = EmailDataset(train_encodings, train_labels)
    eval_dataset = EmailDataset(eval_encodings, eval_labels)

    return train_dataset, eval_dataset
