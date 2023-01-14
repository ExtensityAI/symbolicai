from typing import List
import botdyn as bd


class Demo(bd.Symbol):
    def __init__(self, value = '') -> None:
        super().__init__(value)
    
    @bd.zero_shot(prompt="Generate a random integer between 0 and 10.",
                  constraints=[
                      lambda x: x >= 0,
                      lambda x: x <= 10
                  ])
    def get_random_int(self) -> int:
        pass

    @bd.few_shot(prompt="Generate Japanese names: ",
                 examples=["愛子", "和花", "一郎", "和枝"],
                 limit=2,
                 constraints=[lambda x: len(x) > 1])
    def generate_japanese_names(self) -> list:
        pass
    
    @bd.equals()
    def equals_to(self, other) -> bool:
        pass
    
    @bd.compare(operator='>')
    def larger_than(self, other) -> bool:
        pass
    
    @bd.case(enum=['angry', 'happy', 'annoyed', 'confused', 'satisfied', 'unknown'],
             default='unknown')
    def sentiment_analysis(self, text: str) -> str:
        pass
    
    @bd.translate()
    def translate(self, text: str, language: str) -> str:
        pass

    @bd.zero_shot(prompt="Are this {} names?",
                  default=False,
                  pre_processor=[bd.FormatPromptWithArgs0PreProcessor(),
                                 bd.ArgsToInputPreProcessor(skip=[0])],
                  post_processor=[bd.ConfirmToBoolPostProcessor()])
    def is_name(self, language: str, text: str) -> bool:
        pass
    
    @bd.extract()
    def extract_pattern(self, pattern: str) -> str:
        pass
    
    @bd.clean()
    def clean_text(self, text: str) -> str:
        pass
    
    @bd.summarize()
    def summarize_text(self, text: str) -> str:
        pass
    
    @bd.expression()
    def evaluate_expression(self, expr: str) -> int:
        pass
    
    @bd.simulate()
    def simulate_code(self, code: str) -> str:
        pass
    
    @bd.code()
    def generate_code(self, descr: str) -> str:
        pass
    
    @bd.outline()
    def create_outline(self, text: str) -> str:
        pass
    
    @bd.compose()
    def formulize_text(self, outline: str) -> str:
        pass
    
    @bd.replace()
    def replace_substring(self, replace: str, value: str) -> str:
        pass
    
    @bd.rank()
    def rank_list(self, measure: str, list_: List, order: str) -> str:
        pass
    
    @bd.notify(subscriber={
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
