from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

class RobertaModel:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.model_name = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name)

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
