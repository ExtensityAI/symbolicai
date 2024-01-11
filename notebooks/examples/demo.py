from typing import List

import symai.core as ai


class Demo(ai.Expression):
    def __init__(self, value = '', **kwargs) -> None:
        super().__init__(value, **kwargs)

    @ai.zero_shot(prompt="Generate a random integer between 0 and 10.",
                  constraints=[
                      lambda x: x >= 0,
                      lambda x: x <= 10
                  ])
    def get_random_int(self) -> int:
        pass

    @ai.few_shot(prompt="Generate Japanese names: ",
                 examples=ai.Prompt(["愛子", "和花", "一郎", "和枝"]),
                 limit=2,
                 constraints=[lambda x: len(x) > 1])
    def generate_japanese_names(self) -> list:
        pass

    @ai.equals()
    def equals_to(self, other) -> bool:
        pass

    @ai.compare(operator='>')
    def larger_than(self, other) -> bool:
        pass

    @ai.case(enum=['angry', 'happy', 'annoyed', 'confused', 'satisfied', 'unknown'],
             default='unknown')
    def sentiment_analysis(self, text: str) -> str:
        pass

    @ai.translate()
    def translate(self, text: str, language: str) -> str:
        pass

    @ai.zero_shot(default=False,
                  pre_processors=[ai.ArgsPreProcessor("Are this {} names?"),
                                 ai.ArgsToInputPreProcessor(skip=[0])],
                  post_processors=[ai.ConfirmToBoolPostProcessor()])
    def is_name(self, language: str, text: str) -> bool:
        pass

    @ai.extract()
    def extract_pattern(self, pattern: str) -> str:
        pass

    @ai.clean()
    def clean_text(self, text: str) -> str:
        pass

    @ai.summarize()
    def summarize_text(self, text: str) -> str:
        pass

    @ai.expression()
    def evaluate_expression(self, expr: str) -> int:
        pass

    @ai.simulate()
    def simulate_code(self, code: str) -> str:
        pass

    @ai.code()
    def generate_code(self, descr: str) -> str:
        pass

    @ai.outline()
    def create_outline(self, text: str) -> str:
        pass

    @ai.compose()
    def formulize_text(self, outline: str) -> str:
        pass

    @ai.replace()
    def replace_substring(self, replace: str, value: str) -> str:
        pass

    @ai.rank()
    def rank_list(self, measure: str, list_: List, order: str) -> str:
        pass

    @ai.notify(subscriber={
        'email': lambda x: print('Email sent to ...', x),
        'slack': lambda x: print('Slack message sent to ...', x),
        'fruits': lambda x: Exception('Fruits are not supported yet.')
    })
    def notify_subscriber(self, *args) -> str:
        pass

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return str(self.value)
