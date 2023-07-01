import unittest

from symai import Expression, Import


class TestImport(unittest.TestCase):
    def test_helloworld(self):
        expr = Import('Xpitfire/symhello')
        res = expr('Test')
        self.assertIsNotNone('Hello' in res)

    def test_operational_commands(self):
        Import.install('Xpitfire/symhello')
        self.assertIn('Xpitfire/symhello', Import.list_installed())
        Import.update('Xpitfire/symhello')
        #Import.remove('Xpitfire/symhello')
        #self.assertNotIn('Xpitfire/symhello', Import.list_installed())


if __name__ == '__main__':
    #unittest.main()
    tests = TestImport()
    tests.test_helloworld()
