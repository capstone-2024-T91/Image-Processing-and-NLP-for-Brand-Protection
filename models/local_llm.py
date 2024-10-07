from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.train import train_model
from tqdm import tqdm


class LocalLLM:
    def __init__(self, model_name='distilbert-base-uncased', model_path=None, verbose=False, training=False):
        self.verbose = verbose
        self.model_name = model_name or 'distilbert-base-uncased'
        if training:
            self.model_path = self.model_name  # Load base model
        else:
            self.model_path = model_path or f'./models/{self.model_name}_fine_tuned'
        if self.verbose:
            print(f"Loading tokenizer from: {self.model_path}")
        with tqdm(total=1, desc="Loading Tokenizer", unit="step") as pbar:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            pbar.update(1)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.verbose:
            print(f"Loading model from: {self.model_path}")
        with tqdm(total=1, desc="Loading Model", unit="step") as pbar:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            pbar.update(1)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

    def predict(self, text):
        if self.verbose:
            print("Tokenizing input...")
        inputs = self.tokenizer(text, return_tensors='pt')
        if self.verbose:
            print("Running the model...")
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        if self.verbose:
            print(f"Output probabilities: {probs.detach().numpy()}")
        return torch.argmax(probs) == 1  # Assuming label 1 is 'Phishing'

    def train(self):
        train_model(self.model, self.tokenizer, self.model_name, self.verbose)
