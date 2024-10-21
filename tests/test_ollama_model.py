import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ollama_model import OllamaModel

class TestOllamaModel(unittest.TestCase):
    def setUp(self):
        self.model = OllamaModel(model_name='llama3')

    def test_predict_phishing(self):
        email = "Your account has been suspended. Click here to verify your information."
        result = self.model.predict(email)
        self.assertTrue(result)

    def test_predict_legitimate(self):
        email = "Your invoice is attached. Thank you for your business."
        result = self.model.predict(email)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
