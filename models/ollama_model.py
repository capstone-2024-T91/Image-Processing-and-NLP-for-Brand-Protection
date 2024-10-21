import subprocess
import json

class OllamaModel:
    def __init__(self, model_name='llama3', verbose=False):
        self.model_name = model_name
        self.verbose = verbose

    def predict(self, text):
        # Prepare the prompt
        prompt = f""" Determine whether the following email is a phishing attempt or legitimate.
                    Email:
                    {text}

                    Answer with 'Phishing' or 'Legitimate'.
                    """
        if self.verbose:
            print(f"Sending request to Ollama model '{self.model_name}'...")

        # Run the Ollama command
        try:
            result = subprocess.run(
                ['ollama', 'run', self.model_name,],
                input=prompt.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            # Parse the output
            output = result.stdout.decode('utf-8')
            response_text = output.strip().lower()
            if self.verbose:
                print(f"Ollama model response: {response_text}")
            return 'phishing' in response_text
        except subprocess.CalledProcessError as e:
            print(f"Error running Ollama model: {e.stderr.decode('utf-8')}")
            return False
