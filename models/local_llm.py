from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.train import train_model


class LocalLLM:
    def __init__(self, model_name='distilbert-base-uncased', model_path=None, verbose=False):
        self.verbose = verbose
        self.model_name = model_name
        self.model_path = model_path or f'models/{self.model_name}_fine_tuned'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

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
