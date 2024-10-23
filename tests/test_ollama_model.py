import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ollama_model import OllamaModel
from utils.tester import Tester


class TestOllamaModel(unittest.TestCase):
    def setUp(self):
        model_name = 'llama3'
        self.model = OllamaModel(model_name=model_name)
        self.tester = Tester(
            model=self.model,
            model_name='Ollama-{}'.format(model_name),
            verbose=True
        )

    def test_evaluate(self):
        self.tester.evaluate()

if __name__ == '__main__':
    unittest.main()
