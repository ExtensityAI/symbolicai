from typing import List, Any, Optional, Dict, Callable
from .symbol import Symbol, Expression
from .components import Output
from symai import ai


class ChatBot(Expression):
    _symai_chat: str = """This is a conversation between a chatbot ({}:) and a human (User:). It also includes narration Text (Narrator:) describing the next dialog.
"""
    
    def __init__(self, name: str = 'Symbia', output: Optional[Output] = None):
        super().__init__()
        that = self
        self.name = name
        self.history: List[Symbol] = []
        class CustomInputPreProcessor(ai.ConsoleInputPreProcessor):
            def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
                super().override_reserved_signature_keys(wrp_params, *args, **kwds)
                if f'{name}:' in Symbol(args[0]):
                    input_ = f'\n{str(args[0])}\n$> '
                else:
                    input_ = f'\n{name}: {str(args[0])}\n$> '
                that.history.append(input_)
                return input_
        self._pre_processor = CustomInputPreProcessor
        def custom_post_processor(wrp_self, wrp_params, rsp, *args, **kwargs):
            that.history.append(f'User: {str(rsp)}')
            return rsp
        self._post_processor = custom_post_processor
        
        self.capabilities = {
            'option 1 = [search, web, facts, location, weather, lookup, query]': lambda query, **kwargs: Expression().search(query, **kwargs),
            'option 2 = [fetch, get, crawl, scrape, web]': lambda url, **kwargs: Expression().fetch(url, **kwargs),
            'option 3 = [converse, talk, ask, answer, reply]': lambda context, **kwargs: Expression().query(context, **kwargs),
            'option 4 = [audio, speech, listen': lambda path, **kwargs: Expression().speech(path, **kwargs),
            'option 5 = [scan, read image, ocr, file]': lambda path, **kwargs: Expression().ocr(path, **kwargs),
            'option 6 = [execute, run, code]': lambda code, **kwargs: Expression(code).execute(**kwargs),
            'option 7 = [draw, create meme, generate image]': lambda text, **kwargs: Expression(text).draw(**kwargs),
            'option 8 = [open file, PDF, text file]': lambda path, **kwargs: Expression().open(path, **kwargs),
            'option 9 = [non of the other, unknown, invalid, not understood]': lambda query, **kwargs: self.repeat(query, **kwargs)
        }
        self.capabilities_template = f"""Select the respective group of options that best describes the context of the query.
        Options:
        {self.capabilities}
        Examples:
        When was the first computer invented? =>option 1
        Download the following webpage: https://en.wikipedia.org/wiki/Cat =>option 2
        What do you know about this issue? =>option 3
        Transcribe the following audio file: '/user/home/demo.mp3' =>option 4
        Scan in the text from this image: '/user/home/demo.png' =>option 5
        Run my python code: 'print('hello world')' =>option 6
        Generate a meme with a caption: 'I am batman' =>option 7
        Open my file located at '/user/home/demo.txt =>option 8
        Go to sleep =>option 9
        """
        self.capabilities_choice = ai.Choice(cases=[], # use static context instead
                                             default=list(self.capabilities.keys())[-1])
        
        self.detect_context = [
            'option 1 = [open question, jokes, how are you, chit chat]',
            'option 2 = [specific task or command, query about facts, weather forecast, time, date, location]',
            'option 3 = [exit, quit, bye, goodbye]',
            'option 4 = [help, list of commands, list of capabilities]',
            'option 5 = [follow up question, continuation, more information]',
            'option 6 = [non of the other, unknown, invalid, not understood]'
        ]
        self.detect_context_template = f"""Select the respective group of options that best describes the context of the query.
        Options:
        {self.detect_context}
        Examples:
        How are you feeling today? =>option 1
        Open my file located at '/user/home/demo.txt =>option 2
        Close the program =>option 3
        I don't understand what your functionality is. =>option 4
        Please go on. =>option 5
        I am confused and don't know what to do =>option 6
        """
        self.context_choice = ai.Choice(cases=[], # use static context instead
                                        default=self.detect_context[-1])

    def repeat(self, query, **kwargs):
        return self.narrate('Symbia does not understand and asks to repeat and give more context.', prompt=query)
    
    @property
    def static_context(self) -> str:
        return ChatBot._symai_chat.format(self.name)
    
    def narrate(self, message: str, prompt: str = None, end: bool = False, **kwargs) -> "Symbol":
        narration = f'Narrator: {message}'
        self.history.append(narration)
        attach = Symbol(self.history[-5:])
        @ai.query(context=narration, prompt=prompt, attach=attach, stop=['User:'], **kwargs)
        def _func(_, message) -> str:
            pass
        sym = self._sym_return_type(_func(self, message))
        if end: print(sym)
        return sym
    
    def input(self, message: str = "Please add more information", **kwargs) -> "Symbol":
        # always append User: to the user input
        @ai.userinput(
            pre_processor=[self._pre_processor()],
            post_processor = [ai.StripPostProcessor(), 
                              self._post_processor],
            **kwargs
        )
        def _func(_, message) -> str:
            pass
        return self._sym_return_type(_func(self, message))   
    
    def forward(self, **kwargs):
        pass
            
