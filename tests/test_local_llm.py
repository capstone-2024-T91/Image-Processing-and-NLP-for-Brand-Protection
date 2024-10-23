import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.local_llm import LocalLLM
from utils.tester import Tester


class TestLocalLLM(unittest.TestCase):
    def setUp(self):
        model_name = 'distilbert-base-uncased'
        self.model = LocalLLM()
        self.tester = Tester(
            model=self.model,
            model_name='LocalLLM-{}'.format(model_name),
            verbose=True
        )


    def test_evaluate(self):
        self.tester.evaluate()

if __name__ == '__main__':
    unittest.main()
