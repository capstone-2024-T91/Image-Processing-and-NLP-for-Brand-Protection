import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.roberta_model import RobertaModel
from utils.tester import Tester


class TestRobertaModel(unittest.TestCase):
    def setUp(self):
        self.model = RobertaModel()
        self.tester = Tester(
            model=self.model,
            model_name='RoBERTa',
            verbose=True
        )

    def test_evaluate(self):
        self.tester.evaluate()

if __name__ == '__main__':
    unittest.main()
