from ...models import LLMDataModel
from ...prompts import PromptLanguage, PromptRegistry
from ...strategy import BaseStrategy

registry = PromptRegistry()
registry.register_instruction(PromptLanguage.ENGLISH, "static_context_chain_of_thoughts",
"""
{
    "step_by_step": "Q: Your warehouse has 5 pallets of widgets. You purchase 2 more shipments of widgets. Each shipment contains 3 pallets. How many pallets of widgets do you have now?
A: The warehouse started with 5 pallets of widgets. 2 shipments of 3 pallets each is 6 pallets. 5 pallets + 6 pallets = 11 pallets. The answer is 11 pallets.
Q: Your finance department has $23,000 in the budget. If they allocate $20,000 for a marketing campaign and add $6,000 from other savings, how much is left in the budget?"
    "answer": "The finance department started with $23,000. They allocated $20,000 for a marketing campaign and added $6,000 from other savings. $23,000 - $20,000 + $6,000 = $9,000. The answer is $9,000."
}
"""+
"Think step by step as in the example above before answering."+
"Return the 'step_by_step' and 'answer' properties of the JSON format."
"Always ONLY return a valid JSON format that solve the task.")


class Result(LLMDataModel):
    step_by_step: str
    answer: str


class ChainOfThought(BaseStrategy):
    def __init__(self, data_model=Result, *args, **kwargs):
        super().__init__(data_model=data_model, *args, **kwargs)

    @property
    def task(self):
        task_str = (
            "Think step by step to solve a problem and answer." \
                + r'JSON format: {"step_by_step": "string", "answer": "string"}'
        )
        self.print_verbose(task_str)
        return task_str

    @property
    def static_context(self):
        return registry.instruction("static_context_chain_of_thoughts")
