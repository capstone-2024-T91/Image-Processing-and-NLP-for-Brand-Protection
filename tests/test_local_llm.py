import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.local_llm import LocalLLM

class TestLocalLLM(unittest.TestCase):
    def setUp(self):
        self.model = LocalLLM()

    def test_predict_phishing(self):
        email = "You've won a prize! Click here to claim."
        self.assertTrue(self.model.predict(email))

    def test_predict_legitimate(self):
        email = "Meeting scheduled for tomorrow at 10 AM."
        self.assertFalse(self.model.predict(email))

if __name__ == '__main__':
    unittest.main()
