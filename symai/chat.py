import re
from typing import Any, Optional

from .backend import settings as settings
from .components import (IncludeFilter, InContextClassification, Outline,
                         Output, Sequence)
from .core import *
from .memory import Memory, SlidingWindowListMemory, VectorDatabaseMemory
from .post_processors import ConsolePostProcessor, StripPostProcessor
from .pre_processors import ConsoleInputPreProcessor
from .prompts import SymbiaCapabilities
from .symbol import Expression, Symbol


class ChatBot(Expression):
    _symai_chat: str = """This is a conversation between a chatbot ({}:) and a human (User:). It also includes narration Text (Narrator:) describing the next dialog. The chatbot primarily follows the narrative instructions, and then uses the user input to condition on for the generated response.\n"""

    def __init__(self, value = None, name: str = 'Symbia', output: Optional[Output] = None, verbose: bool = False):
        super().__init__(value)
        self.verbose: bool        = verbose
        self.name                 = name
        self.last_user_input: str = ''

        self.memory:           Memory = SlidingWindowListMemory()
        self.long_term_memory: Memory = VectorDatabaseMemory(enabled=settings.SYMAI_CONFIG['INDEXING_ENGINE_API_KEY'] is not None)

        self._preprocessor  = ChatBot._init_custom_input_preprocessor(name=name, that=self)
        self._postprocessor = ChatBot._init_custom_input_postprocessor(that=self)

        self.classify = InContextClassification(SymbiaCapabilities())
        Expression.command(engines=['symbolic'], expression_engine='wolframalpha')

    def repeat(self, query, **kwargs):
        return self.narrate('Symbia does not understand and asks to repeat and give more context.', prompt=query)

    def narrate(self, message: str, context: str = None, end: bool = False, do_recall: bool = True, **kwargs) -> Symbol:
        narration = f'Narrator: {message}'
        self.memory.store(narration)

        ctxt    = context if context is not None else ''
        value   = f"{self.static_context}ADDITIONAL FACTS AND CONTEXT\n{ctxt}\n" if ctxt is not None else ''
        value  += 'SHORT-TERM MEMORY RECALL\n'
        value  += '\n'.join(self.memory.recall()) # TODO: use vector search DB
        value  += '\n\nLONG-TERM MEMORY RECALL (Consider only if relevant to the user query!)\n\n'
        query   = f'{self.last_user_input}\n{message}\n\n'
        recall  = self.long_term_memory.recall(query) if do_recall else []
        value  += '\n'.join(recall)
        value  += f'\n{self.name}:'

        if self.verbose: print('[DEBUG] long-term memory recall: ', recall)
        if self.verbose: print('[DEBUG] narration: ', narration)
        if self.verbose: print('[DEBUG] function context: ', ctxt)

        @zero_shot(prompt=value, stop=['User:'], **kwargs)
        def _func(_) -> str:
            pass
        model_rsp = _func(self)

        rsp = f"{self.name}: {model_rsp}"
        if self.verbose: print('[DEBUG] model reply: ', model_rsp)
        memory = f"{self.last_user_input}" if do_recall else ""
        if len(memory) > 0: self.long_term_memory.store(memory)
        memory = f"{model_rsp}" if do_recall else ""
        if len(memory) > 0: self.long_term_memory.store(memory)
        if self.verbose: print('[DEBUG] store new memory reply: ', memory)

        sym = Symbol(rsp)

        if end: print(sym)

        return sym

    def input(self, message: str = "Please add more information", **kwargs) -> Symbol:
        @userinput(
            pre_processor=[self._preprocessor()],
            post_processor=[StripPostProcessor(),
                            self._postprocessor()],
            **kwargs
        )
        def _func(_, message) -> str:
            pass

        return Symbol(_func(self, message))

    @property
    def static_context(self) -> str:
        return ChatBot._symai_chat.format(self.name)

    @property
    def _sym_return_type(self):
        return ChatBot

    @staticmethod
    def _init_custom_input_preprocessor(name, that):
        class CustomInputPreProcessor(ConsoleInputPreProcessor):
            def __call__(self, wrp_self, wrp_params, *args: Any, **kwargs: Any) -> Any:
                super().override_reserved_signature_keys(wrp_params, *args, **kwargs)

                if f'{name}:' in Symbol(args[0]): input_ = f'\n{str(args[0])}\n$> '
                else:                             input_ = f'\n{name}: {str(args[0])}\n$> '

                that.memory.store(input_)
                that.long_term_memory.store(str(args[0]))

                return input_

        return CustomInputPreProcessor

    @staticmethod
    def _init_custom_input_postprocessor(that):
        class CustomInputPostProcessor(ConsolePostProcessor):
            def __call__(self, wrp_self, wrp_params, rsp, *args, **kwargs):
                that.memory.store(f'User: {str(rsp)}')

                return rsp

        return CustomInputPostProcessor

class SymbiaChat(ChatBot):
    def forward(self):
        message = self.narrate(f'{self.name} introduces herself, writes a greeting message and asks how to help.', do_recall=False)

        while True:
            usr = self.input(message)
            self.last_user_input = usr
            if self.verbose: print('[DEBUG] user query: ', usr)

            if len(str(usr)) > 0:
                ctxt = str(self.classify(usr))
            else:
                ctxt = '[DK]'

            if self.verbose: print('[DEBUG] context: ', ctxt)

            if '[EXIT]' in ctxt:
                self.narrate(f'{self.name} writes goodbye message.', end=True, do_recall=False)
                break

            elif '[HELP]' in ctxt:
                thought = self._extract_thought(ctxt)
                message = self.narrate(f'{self.name} ', context=thought, do_recall=False)

            elif '[DK]' in ctxt:
                thought = self._extract_thought(ctxt)
                q = usr.extract('user query request')
                rsp = self.search(q)
                message = self.narrate(f'{self.name} is not sure about the message and references to search results.', context=thought, payload=rsp)

            else:
                try:
                    if '[SYMBOLIC]' in ctxt:
                        q = usr.extract("mathematical formula that WolframAlpha can solve")
                        rsp = q.expression()
                        thought = self._extract_thought(ctxt)
                        message = self.narrate(f'{self.name} replies to the user and provides the solution of the math problem.', context=rsp)

                    if '[SEARCH]' in ctxt:
                        q = usr.extract('user query request')
                        rsp = self.search(q)
                        thought = self._extract_thought(ctxt)
                        message = self.narrate(f'{self.name} replies to the user based on the online search results.', context=rsp)

                    elif '[CRAWLER]' in ctxt:
                        q = usr.extract('URL from text')
                        q = q.convert('proper URL, example: https://www.google.com')
                        site = self.fetch(q)
                        site.save('tmp.html')
                        thought = self._extract_thought(ctxt)
                        message = self.narrate(f'{self.name} explains that the website is downloaded to the `tmp.html` file.')

                    elif '[SPEECH-TO-TEXT]' in ctxt:
                        q = usr.extract('extract file path')
                        rsp = self.speech(q)
                        thought = self._extract_thought(ctxt)
                        message = self.narrate('Symbia replies to the user and transcribes the content of the audio file.', context=rsp)

                    elif '[TEXT-TO-IMAGE]' in ctxt:
                        q = usr.extract('text for image creation')
                        rsp = q.draw()
                        thought = self._extract_thought(ctxt)
                        message = self.narrate('Symbia replies to the user and provides the image URL.', context=rsp)

                    elif '[OCR]' in ctxt:
                        url = usr.extract('extract url')
                        rsp = Expression().ocr(url)
                        thought = self._extract_thought(ctxt)
                        message = self.narrate('Symbia replies to the user and provides OCR text from the image.',
                                                context=rsp)

                    elif '[RETRIEVAL]' in ctxt:
                        file = usr.extract('extract file path')
                        q = usr.extract('user question')
                        rsp = file.fstream(
                            Sequence(
                                IncludeFilter('include only facts related to the user question: ' @ q),
                                Outline()
                            )
                        )
                        thought = self._extract_thought(ctxt)
                        message = self.narrate('Symbia replies to the user and outlines and relies to the user query.', context=rsp)

                    else:
                        q = usr.extract('user query request')
                        rsp = self.search(q)
                        thought = self._extract_thought(ctxt)
                        message = self.narrate('Symbia tries to interpret the response, and if unclear asks the user to restate the statement or add more context.', context=thought)

                except Exception as e:
                    thought = self._extract_thought(ctxt)
                    message = self.narrate('Symbia apologizes and explains the user what went wrong.', context=str(e), do_recall=False)

    def _extract_thought(self, msg: str) -> str:
        return re.findall(r'\{([^}]+)\}', msg).pop()

def run() -> None:
    chat = SymbiaChat()
    chat()

if __name__ == '__main__':
    run()
