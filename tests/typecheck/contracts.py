from typing import assert_type

from symai.contract.contract import Contract, ContractResult
from symai.contract.models import LLMDataModel
from symai.runtime.runtime import LanguageModel


class Input(LLMDataModel):
    text: str


class Output(LLMDataModel):
    answer: str


contract = Contract(
    instruction="Answer.",
    input_type=Input,
    output_type=Output,
)
assert_type(contract, Contract[Input, Output])


def prove_contract_result_inference(engine: LanguageModel, input_value: Input) -> None:
    assert_type(contract(engine, input_value), Output)
    assert_type(contract.run(engine, input_value), ContractResult[Output])
