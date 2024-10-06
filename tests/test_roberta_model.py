import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.roberta_model import RobertaModel

class TestRobertaModel(unittest.TestCase):
    def setUp(self):
        self.model = RobertaModel()

    def test_predict_phishing(self):
        email = "Your account has been compromised. Reset your password now."
        self.assertTrue(self.model.predict(email))

    def test_predict_legitimate(self):
        email = "Thank you for your purchase. Your order will arrive soon."
        self.assertFalse(self.model.predict(email))

if __name__ == '__main__':
    unittest.main()
