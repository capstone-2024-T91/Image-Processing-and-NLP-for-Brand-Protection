import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.local_llm import LocalLLM
from models.roberta_model import RobertaModel
from utils.preprocess import preprocess_email

def parse_arguments():
    parser = argparse.ArgumentParser(description='Phishing Email Detector')
    parser.add_argument('email_text', type=str, help='The email text to analyze')
    parser.add_argument('-local', action='store_true', help='Use the local LLM')
    parser.add_argument('-r', action='store_true', help='Use the RoBERTa model')
    parser.add_argument('-v', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def main():
    args = parse_arguments()
    email_text = args.email_text
    verbose = args.v

    # Preprocess the email
    processed_text = preprocess_email(email_text)
    if verbose:
        print("Preprocessed Email Text:")
        print(processed_text)

    # Select the model
    if args.local:
        if verbose:
            print("Using Local LLM for prediction.")
        model = LocalLLM(verbose=verbose)
    elif args.r:
        if verbose:
            print("Using RoBERTa model for prediction.")
        model = RobertaModel(verbose=verbose)
    else:
        print("No model specified. Use -local or -r to select a model.")
        return

    # Predict
    prediction = model.predict(processed_text)
    print(f"\nPrediction: {'Phishing Email' if prediction else 'Legitimate Email'}")

if __name__ == '__main__':
    main()
