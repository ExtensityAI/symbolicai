import unittest

from symai import Expression, Interface


class TestInterface(unittest.TestCase):
    def test_dalle(self):
        expr = Interface('dall-e')
        res = expr('a cat with a hat')
        self.assertIsNotNone('http' in res)

    def test_google(self):
        expr = Interface('serpapi')
        res = expr('Who is Barack Obama?')
        self.assertIsNotNone('president' in res)

    def test_file(self):
        expr = Interface('file')
        res = expr('./LICENSE')
        self.assertTrue('Copyright (c)' in res, res)

    def test_console(self):
        expr = Interface('console')
        res = expr('Hallo Welt!')

    def test_input(self):
        expr = Interface('input')
        res = expr()
        self.assertTrue('Hallo Welt!' in res)

    def test_whisper(self):
        expr = Interface('whisper')
        res = expr('examples/audio.mp3')
        self.assertTrue(res == 'I may have overslept.')

    def test_selenium(self):
        expr = Interface('selenium')
        res = expr(url='https://en.wikipedia.org/wiki/Logic_programming')
        self.assertTrue('programming' in str(res), res)

    def test_clip(self):
        expr = Interface('clip')
        res = expr('https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/cute-cat-photos-1593441022.jpg',
                   ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])
        res = res.argmax()
        self.assertTrue(0 == res)

    def test_ocr(self):
        expr = Interface('ocr')
        res = expr('https://media-cdn.tripadvisor.com/media/photo-p/0f/da/22/3a/rechnung.jpg')
        self.assertTrue('China' in res)

    def test_pinecone(self):
        expr = Interface('pinecone')
        expr(Expression('Hello World!').zip(), operation='add')
        expr(Expression('I like cookies!').zip(), operation='add')
        res = expr(Expression('hello').embed().value, operation='search').ast()
        self.assertTrue('Hello' in str(res['matches'][0]['metadata']['text']), res)

    def test_wolframalpha(self):
        expr = Interface('wolframalpha')
        res = expr('x^2 + 2x + 1, x = 4')
        self.assertTrue(res == 25, res)

    def test_python(self):
        expr = Interface('python')
        res = expr('x = 5')
        self.assertTrue(res is not None, res)

    def test_blip2(self):
        caption = Interface('blip-2')
        res = caption(r"https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
                'a photography of')
        self.assertTrue(res is not None, res)


if __name__ == '__main__':
    unittest.main()
