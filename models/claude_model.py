import anthropic
import os
import load_dotenv

load_dotenv.load_dotenv()

class ClaudeModel:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set the 'ANTHROPIC_API_KEY' environment variable.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def predict(self, text):
        prompt = f"Determine whether the following email is a phishing attempt or legitimate:\n\nEmail: {text}\n\nAnswer with 'Phishing' or 'Legitimate' ONLY answer with 'Phishing' if phishing or 'Legitimate' if legitimate. If you are unsure, answer 'Legitimate'."
        if self.verbose:
            print("Sending request to Anthropic API...")

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        result = response.content[0].text.strip().lower()
        if self.verbose:
            print(f"Anthropic API response: {result}")
        return result == 'phishing'
