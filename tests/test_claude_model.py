import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.claude_model import ClaudeModel
from utils.tester import Tester


class TestClaudeModel(unittest.TestCase):
    def setUp(self):
        self.model = ClaudeModel()
        self.tester = Tester(
            model=self.model,
            model_name='Claude',
            verbose=True
        )

    def test_evaluate(self):
        self.tester.evaluate()

if __name__ == '__main__':
        unittest.main()
