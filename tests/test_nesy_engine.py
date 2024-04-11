import unittest
from pathlib import Path

from symai import Symbol


class TestNesyEngine(unittest.TestCase):
    def test_init(self):
        x = Symbol('This is a test!')
        x.query('What is this?')

        # if no errors are raised, then the test is successful
        self.assertTrue(True)

    def test_vision_component(self):
        file = Path(__file__).parent.parent / 'assets' / 'images' / 'cat.jpg'
        x = Symbol(f'<<vision:{file}:>>')
        res = x.query('What is in the image?')

        # it makes sense here to explicitly check if there is a cat; we are testing the vision component
        # which only gpt-4-* models can handle and they should be able to detect a cat in the image
        self.assertTrue('cat' in res.value)

        file = 'https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/cat.jpg'
        x = Symbol(f'<<vision:{file}:>>')
        res = x.query('What is in the image?')

        # same check but for url
        self.assertTrue('cat' in res.value)


if __name__ == '__main__':
    unittest.main()
