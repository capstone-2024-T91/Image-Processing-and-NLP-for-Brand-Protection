import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.openai_model import OpenAIModel

class TestOpenAIModel(unittest.TestCase):
    def setUp(self):
        self.model = OpenAIModel()

    def test_predict_phishing(self):
        email = "Your account has been suspended. Click here to verify your information."
        self.assertTrue(self.model.predict(email))

    def test_predict_legitimate(self):
        email = "Hi, my name is John. I'm following up on the email I sent you last week."
        self.assertFalse(self.model.predict(email))

if __name__ == '__main__':
    unittest.main()
