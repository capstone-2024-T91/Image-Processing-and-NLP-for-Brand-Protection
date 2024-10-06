from openai import OpenAI
import os
import load_dotenv

load_dotenv.load_dotenv()

class OpenAIModel:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI()
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set the 'OPENAI_API_KEY' environment variable.")

    def predict(self, text):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a hyper specialized agent that helps people determine whether an email is a phishing attempt or legitimate."},
                {
                    "role": "user",
                    "content": f"Determine whether the following email is a phishing attempt or legitimate:\n\nEmail: {text}\n\nAnswer with 'Phishing' if phishing or 'safe' if legitimate."
                }
            ]
        )
        if self.verbose:
            print("Sending request to OpenAI API...")
        result = response.choices[0].message.content.strip().lower()
        if self.verbose:
            print(f"OpenAI API response: {result}")
        print(result)
        return 'phishing' in result
