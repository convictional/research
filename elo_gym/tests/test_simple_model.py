import unittest
from models.simple_model import SimpleModel

# Run BASH: python -m unittest -v tests.test_simple_model


class TestSimpleModel(unittest.TestCase):
    def test_simple_model_process_prompt(self):
        model = SimpleModel()
        prompt = "What is the capital of France?"
        response = model.process_prompt(prompt)
        self.assertIsInstance(response, dict)
        self.assertIn("response", response)
        self.assertEqual(response["response"], f"Simple Model Response: {prompt}")


if __name__ == "__main__":
    unittest.main()
