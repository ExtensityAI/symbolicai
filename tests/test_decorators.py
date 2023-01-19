import os
# for debugging
# attention this constantly overwrites the keys config file
#os.environ['OPENAI_API_KEY'] = ''

import unittest
from examples.demo import Demo


class TestDecorator(unittest.TestCase):
    def test_few_shot(self):
        demo = Demo()
        names = demo.generate_japanese_names()
        self.assertIsNotNone(names)
        self.assertTrue(len(names) == 2)
        print(names)
        res = demo.is_name('Japanese', names)
        self.assertTrue(res)
        res = demo.is_name('German', names)
        self.assertFalse(res)

    def test_zero_shot(self):
        demo = Demo()
        val = demo.get_random_int()
        self.assertIsNotNone(val)
        print(val)
        
    def test_equals(self):
        demo = Demo(2)
        res = demo.equals_to('2')
        self.assertTrue(res)
        
    def test_compare(self):
        demo = Demo(175)
        res = demo.larger_than(66)
        self.assertTrue(res)
        
    def test_rank(self):
        demo = Demo(['1.66m', '1.75m', '1.80m'])
        res = demo.rank_list('hight', ['1.75m', '1.66m', '1.80m'], order='asc')
        self.assertTrue(demo.equals_to(res))
        
    def test_case(self):
        demo = Demo('angry')
        res = demo.sentiment_analysis('I really hate this stupid application because it does not work.')
        self.assertTrue(demo.equals_to(res))
        
    def test_translate(self):
        demo = Demo()
        res = demo.translate('I feel tired today.', language='Spanish')
        self.assertIsNotNone(res)
        
    def test_extract_pattern(self):
        demo = Demo('Open the settings.json file, edit the env property and run the application again with the following command: python main.py')
        res = demo.extract_pattern('Files with *.json')
        self.assertTrue(res == 'settings.json')
        
    def test_replace(self):
        demo = Demo('Steve Ballmer is the CEO of Microsoft.')
        res = demo.replace_substring('Steve Ballmer', 'Satya Nadella')
        self.assertTrue('Satya' in res)

    def test_expression(self):
        demo = Demo(18)
        res = demo.evaluate_expression('2 + 4 * 2 ^ 2')
        self.assertTrue(demo.value == res)
        
    def test_notify_subscriber(self):
        demo = Demo()
        res = demo.notify_subscriber('You can contact us via email at office@alphacore.eu', 
                                     subscriber={'europe': lambda x: Exception('Not allowed')})
        self.assertTrue('email' in res)


if __name__ == '__main__':
    unittest.main()
