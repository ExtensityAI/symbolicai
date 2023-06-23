"""
Evaluation script for the NeuroSymbolic AI project containing a number of different test cases.

Before using a local engine, make sure to run:
```
    export TRANSFORMERS_CACHE="/system/user/publicwork/schmied/symai/huggingface"
    symsvr
    symclient
```

"""

import argparse
from pathlib import Path

import pandas as pd

import symai as ai
from symai import Expression, Symbol
from symai.backend.engine_nesy_client import NeSyClientEngine


def setup_engine():
    print("Initializing engine...")
    engine = NeSyClientEngine()
    Expression.setup(engines={'neurosymbolic': engine})
    return engine


class Evaluation:
    """
    Evaluation class for the NeuroSymbolic AI project.
    Conducts a series of logic and expression tests and saves the results to a specified directory.

    """
    def __init__(self) -> None:
        self.math_tests = [
            {"test": "Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.", "expected": 4},
            {"test": "Let x(g) = 9*g + 1. Let q(c) = 2*c + 1. Let f(i) = 3*i - 39. Let w(j) = q(x(j)). Calculate f(w(a)).", "expected": "54*a - 30"},
            {"test": "Let e(l) = l - 6. Is 2 a factor of both e(9) and 2?",  "expected": "False"},
            {"test": "Let u(n) = -n**3 - n**2. Let e(c) = -2*c**3 + c. Let l(j) = -118*e(j) + 54*u(j). What is the derivative of l(a)?",  "expected": "546*a**2 - 108*a - 118"},
            {"test": "Three letters picked without replacement from qqqkkklkqkkk. Give prob of sequence qql.",  "expected": "1/110"},
        ]

    def evaluate(self, save_dir=None):
        # evalutate tests
        df_logic = self.evaluate_logic_tests()
        df_expression = self.evaluate_expression_tests()
        df_mathematics = self.evaluate_mathematics_tests()

        # combine results
        df_results = pd.concat([df_logic, df_expression, df_mathematics])
        df_success = df_results.copy()
        df_success["Success"] = df_success["Success"].astype(int)
        df_success = df_success.groupby("Kind")["Success"].describe()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(save_dir / "results.csv", index=False)
            df_success.round(3).to_csv(save_dir / "success.csv")

        return df_results

    def evaluate_logic_tests(self):
        eval_dict = []
        tests = [
            {"res": Symbol('the horn only sounds on Sundays') & Symbol('I hear the horn'), "expected": "it is Sunday"},
            {"res": Symbol('the horn sounds on Sundays') | Symbol('the horn sounds on Mondays'), "expected": "Sundays or Mondays"},
            {"res": Symbol('The duck quaks.') ^ Symbol('The duck does not quak.'), "expected": "The duck quaks."},
            {"res": Symbol('A line parallel to y = 4x + 6 passes through (5, 10). What is the y-coordinate of the point where this line crosses the y-axis?'), "expected": "-10"},
            {"res": Symbol("Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother."), "expected": "Jay's brother is John."},
            {"res": Symbol('is 1000 bigger than 1063.472?'), "expected": "No"},
            {"res": Symbol('John weights 85 pounds. Jeff weighs 105 pounds. Jake weighs 115 pounds. Two of them standing together on the same scale could weigh 200 pounds.'), "expected": "Yes. John and Jake could weigh 200 pounds together."},
        ]
        for test in tests:
            eval_dict.append({
                "Kind": "logic",
                "Result": str(test["res"]),
                "Expected": test["expected"],
                "Success": test["expected"] in test["res"]
            })

        return pd.DataFrame(eval_dict)

    def evaluate_logic_tests_chained(self):
        pass

    def evaluate_expression_tests(self):
        eval_dict = []
        tests = [
            {"res": Symbol(1).expression('self + 2'), "expected": 3},
            {"res": Symbol(2).expression('2 ^ self'), "expected": 4}
        ]
        for test in tests:
            eval_dict.append({
                "Kind": "expression",
                "Result": test["res"],
                "Expected": test["expected"],
                "Success": test["expected"] == test["res"]
            })

        return pd.DataFrame(eval_dict)

    def evaluate_expression_tests_chained(self):
        pass

    def evaluate_mathematics_tests(self):
        # examples from: https://github.com/deepmind/mathematics_dataset
        eval_dict = []
        for test in self.math_tests:
            res = Symbol(test["test"])
            eval_dict.append({
                "Kind": "mathematics",
                "Result": str(res),
                "Expected": test["expected"],
                "Success": test["expected"] in res
            })
        return pd.DataFrame(eval_dict)

    def evaluate_mathematics_tests_chained(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./evaluation/results")
    args = parser.parse_args()

    # setup engine
    engine = setup_engine()

    # run evaluation
    print("Running evaluation...")
    evaluation = Evaluation()
    df_results = evaluation.evaluate(save_dir=args.save_dir)
    print(df_results)



