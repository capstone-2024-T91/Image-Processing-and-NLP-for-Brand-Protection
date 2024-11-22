from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from models.local_llm import LocalLLM
from models.roberta_model import RobertaModel
from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from models.ollama_model import OllamaModel
from utils.preprocess import preprocess_email

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Preload models
models = {}

def initialize_models():
    global models
    verbose = True  # Set to False if verbose logging is not needed
    models['local'] = LocalLLM(verbose=verbose)
    models['roberta'] = RobertaModel(verbose=verbose)
    models['openai'] = OpenAIModel(verbose=verbose)
    models['claude'] = ClaudeModel(verbose=verbose)
    models['ollama'] = OllamaModel(verbose=verbose)
    print("All models preloaded successfully.")

@app.route('/detect_phishing', methods=['GET'])
def detect_phishing():
    email_text = request.args.get('email_text')
    model_option = request.args.get('model_option', 'local')  # Default to 'local'

    if not email_text:
        return jsonify({"error": "Please provide 'email_text' in the query parameters."}), 400

    if model_option not in models:
        return jsonify({"error": f"Model '{model_option}' is not available. Choose from {list(models.keys())}."}), 400

    try:
        # Preprocess the email text
        processed_text = preprocess_email(email_text)
        
        # Get the preloaded model
        model = models[model_option]

        # Predict
        prediction = model.predict(processed_text)
        result = "Phishing Email" if prediction else "Legitimate Email"
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "An error occurred during prediction.",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Initialize models before starting the server
    initialize_models()
    app.run(debug=True, host='0.0.0.0')
