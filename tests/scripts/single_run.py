import argparse
from pathlib import Path

import pandas as pd

import symai as ai
from symai import Expression, Symbol
from symai.backend.engine_nesy_client import NeSyClientEngine
from symai.extended import Solver


def setup_engine():
    print("Initializing engine...")
    engine = NeSyClientEngine()
    Expression.setup(engines={'neurosymbolic': engine})
    return engine


def run_single_test():
    tests = [
        {"test": 'Add the square root of 5 to the square root of x to the power of 2, where x equals 10', "expected": "421.26827"},
        {"test": '4 + 433.43 + e^0.38877 - 33.101 / pi', "expected": "428.369"},
        {"test": 'John weights 85 pounds. Jeff weighs 105 pounds. Jake weighs 115 pounds. Two of them standing together on the same scale could weigh 200 pounds.', "expected": "Yes. John and Jake could weigh 200 pounds together."},
        {"test": "Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.", "expected": 4},
        {"test": 'A line parallel to y = 4x + 6 passes through (5, 10). What is the y-coordinate of the point where this line crosses the y-axis?', "expected": "-10"},
    ]

    solver = Solver()
    for test in tests:
        res = solver(test["test"])
        print(res)
        #assert res == test["expected"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./evaluation/results")
    args = parser.parse_args()

    # setup engine
    #engine = setup_engine()

    # run evaluation
    print("Running evaluation...")
    df_results = run_single_test()
    print(df_results)



