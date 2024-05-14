import unittest

from symai import Import


class TestImport(unittest.TestCase):
    def test_helloworld(self):
        expr = Import('ExtensityAI/rickshell')
        res = expr('Test')
        self.assertIsNotNone(res)

    def test_operational_commands(self):
        Import.install('ExtensityAI/symask')
        self.assertIn('ExtensityAI/symask', Import.list_installed())
        Import.update('ExtensityAI/symask')


if __name__ == '__main__':
    #unittest.main()
    tests = TestImport()
    tests.test_helloworld()
