import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.openai_model import OpenAIModel
from utils.tester import Tester


class TestOpenAIModel(unittest.TestCase):
    def setUp(self):
        self.model = OpenAIModel()
        self.tester = Tester(
            model=self.model,
            model_name='OpenAI GPT',
            verbose=True
        )

    def test_evaluate(self):
        self.tester.evaluate()

if __name__ == '__main__':
    unittest.main()
