import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.claude_model import ClaudeModel

class TestClaudeModel(unittest.TestCase):
    def setUp(self):
        self.model = ClaudeModel()

    def test_predict_phishing(self):
        email = "We've detected unusual activity on your account. Reset your password now."
        self.assertTrue(self.model.predict(email))

    def test_predict_legitimate(self):
        email = "Please find the meeting agenda attached."
        self.assertFalse(self.model.predict(email))

if __name__ == '__main__':
        unittest.main()
