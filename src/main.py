import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.local_llm import LocalLLM
from models.roberta_model import RobertaModel
from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from models.ollama_model import OllamaModel
from utils.preprocess import preprocess_email


def parse_arguments():
    parser = argparse.ArgumentParser(description='Phishing Email Detector')
    parser.add_argument('email_text', nargs='?', default='', type=str, help='The email text to analyze')
    parser.add_argument('-local', type=str, help='Specify a local model by name')
    parser.add_argument('-llm', action='store_true', help='Use the default local LLM')
    parser.add_argument('-r', action='store_true', help='Use the RoBERTa model')
    parser.add_argument('-openai', action='store_true', help='Use OpenAI GPT models')
    parser.add_argument('-claude', action='store_true', help='Use Anthropic Claude models')
    parser.add_argument('-o', type=str, help='Use an Ollama model (specify the model name)')
    parser.add_argument('-train', action='store_true', help='Train the selected model')
    parser.add_argument('-v', action='store_true', help='Enable verbose output')
    parser.add_argument('-model_path', type=str, help='Path to the fine-tuned model directory')
    parser.add_argument('-checkpoint', type=str, help='Path to the training checkpoint')
    parser.add_argument('-run_id', type=str, help='W&B run ID for resuming training')
    return parser.parse_args()

def main():
    args = parse_arguments()
    verbose = args.v
    model_path = args.model_path
    if args.checkpoint and not args.train:
        print("Checkpoint is only used during training. Use -train to train the model.")
        return
    
    checkpoint = args.checkpoint if args.checkpoint else None
    run_id = args.run_id if args.run_id else None

    if args.train:
        # Training Mode
        if args.local or args.llm:
            model = LocalLLM(model_name=args.local, verbose=verbose, training=True, checkpoint=checkpoint, run_id=run_id)
        elif args.r:
            model = RobertaModel(verbose=verbose, training=True, checkpoint=checkpoint, run_id=run_id)
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
        model = LocalLLM(model_name=args.local, model_path=model_path, verbose=verbose)
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
    elif args.o:
        if verbose:
            print(f"Using Ollama model '{args.o}' for prediction.")
        model = OllamaModel(model_name=args.o, verbose=verbose)
    else:
        print("No model specified. Use -local, -llm, -r, -openai, -o, or -claude to select a model.")
        return

    # Predict
    prediction = model.predict(processed_text)
    print(f"\nPrediction: {'Phishing Email' if prediction else 'Legitimate Email'}")

if __name__ == '__main__':
    main()
