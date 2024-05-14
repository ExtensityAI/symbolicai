import unittest

from symai import Expression

Expression.command(time_clock=True)


class TestFineTuning(unittest.TestCase):
    def test_fine_tune_prepare_data(self):
        expr = Expression()
        res = expr.tune(operation='prepare_data',
                        file='examples/dbpedia_samples.jsonl')
        print(res)

    def test_fine_tune_create(self):
        expr = Expression()
        res = expr.tune(operation='create',
                        train_file='examples/dbpedia_samples_prepared_train.jsonl',
                        valid_file='examples/dbpedia_samples_prepared_eval.jsonl')
        print(res['id'])

    def test_fine_tune_get(self):
        expr = Expression()
        res = expr.tune(operation='get', id='ft-Gdh2pHtkM0Pqihyw3euoySrS')
        print(res['status'])

    def test_fine_tune_cancel(self):
        expr = Expression()
        res = expr.tune(operation='cancel', id='ft-Gdh2pHtkM0Pqihyw3euoySrS')
        print(res['status'])

    def test_fine_tune_delete(self):
        expr = Expression()
        res = expr.tune(operation='delete', id='ft-Gdh2pHtkM0Pqihyw3euoySrS')
        print(res)

    def test_fine_tune_results(self):
        expr = Expression()
        res = expr.tune(operation='results', id='ft-Gdh2pHtkM0Pqihyw3euoySrS')
        print(res)

    def test_fine_tune_list(self):
        expr = Expression()
        res = expr.tune(operation='list')
        print(res)


if __name__ == '__main__':
    unittest.main()
