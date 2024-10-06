import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.local_llm import LocalLLM
from models.roberta_model import RobertaModel
from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from utils.preprocess import preprocess_email


def parse_arguments():
    parser = argparse.ArgumentParser(description='Phishing Email Detector')
    parser.add_argument('email_text', nargs='?', default='', type=str, help='The email text to analyze')
    parser.add_argument('-llm', type=str, help='Specify a local LLM by name')
    parser.add_argument('-local', action='store_true', help='Use the default local LLM')
    parser.add_argument('-r', action='store_true', help='Use the RoBERTa model')
    parser.add_argument('-openai', action='store_true', help='Use OpenAI GPT models')
    parser.add_argument('-claude', action='store_true', help='Use Anthropic Claude models')
    parser.add_argument('-train', action='store_true', help='Train the selected model')
    parser.add_argument('-v', action='store_true', help='Enable verbose output')
    parser.add_argument('-model_path', type=str, help='Path to the fine-tuned model directory')
    return parser.parse_args()

def main():
    args = parse_arguments()
    verbose = args.v
    model_path = args.model_path

    if args.train:
        # Training Mode
        if args.local or args.llm:
            model = LocalLLM(model_name=args.llm, verbose=verbose)
        elif args.r:
            model = RobertaModel(verbose=verbose)
        else:
            print("Please specify a model to train using -local, -llm, or -r.")
            return
        model.train()
        return

    if not args.email_text:
        print("Please provide email text for analysis.")
        return

    email_text = args.email_text

    # Preprocess the email
    processed_text = preprocess_email(email_text)
    if verbose:
        print("Preprocessed Email Text:")
        print(processed_text)

    # Select the model
    if args.local or args.llm:
        if verbose:
            print("Using Local LLM for prediction.")
        model = LocalLLM(model_name=args.llm, model_path=model_path, verbose=verbose)
    elif args.r:
        if verbose:
            print("Using RoBERTa model for prediction.")
        model = RobertaModel(model_path=model_path, verbose=verbose)
    elif args.openai:
        if verbose:
            print("Using OpenAI GPT model for prediction.")
        model = OpenAIModel(verbose=verbose)
    elif args.claude:
        if verbose:
            print("Using Anthropic Claude model for prediction.")
        model = ClaudeModel(verbose=verbose)
    else:
        print("No model specified. Use -local, -llm, -r, -openai, or -claude to select a model.")
        return

    # Predict
    prediction = model.predict(processed_text)
    print(f"\nPrediction: {'Phishing Email' if prediction else 'Legitimate Email'}")

if __name__ == '__main__':
    main()
