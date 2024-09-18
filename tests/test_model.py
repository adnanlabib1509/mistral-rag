import unittest
from src.model import MistralModel

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MistralModel()

    def test_generate(self):
        prompt = "Translate the following English text to French: 'Hello, how are you?'"
        response = self.model.generate(prompt)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

if __name__ == '__main__':
    unittest.main()