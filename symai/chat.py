import logging
import re

from typing import Any, Optional

from .backend.mixin.tracker import OpenAICostTracker
from .backend import settings as settings
from .components import (IncludeFilter, InContextClassification,
                         Outline, Output, Sequence)
from .memory import Memory, SlidingWindowListMemory, VectorDatabaseMemory
from .post_processors import ConsolePostProcessor, StripPostProcessor
from .pre_processors import ConsoleInputPreProcessor
from .prompts import MemoryCapabilities, SymbiaCapabilities
from .symbol import Expression, Symbol
from .interfaces import Interface
from . import core

logging.getLogger('charset_normalizer').setLevel(logging.ERROR)


class ChatBot(Expression):
    _symai_chat: str = '''This is a conversation between a chatbot (Symbia:) and a human (User:). The chatbot follows a narrative structure, primarily relying on the provided instructions. It uses the user's input as a conditioning factor to generate its responses. Whenever Symbia retrieves any long-term memories, it checks the user's query and incorporates information from the long-term memory buffer into its response. If the long-term memories cannot provide a suitable answer, Symbia then checks its short-term memory to be aware of the topics discussed in the recent conversation rounds. Your primary task is to reply to the user's question or statement by generating a relevant and contextually appropriate response. Do not focus on filling the scratchpad with narration, long-term memory recall, short-term memory recall, or reflections. Always consider any follow-up questions or relevant information from the user to generate a response that is contextually relevant. Endeavor to reply to the greatest possible effort.'''

    def __init__(self, value = None, name: str = 'Symbia', output: Optional[Output] = None, verbose: bool = False, **kwargs):
        super().__init__(value, **kwargs)
        self.sym_return_type = ChatBot
        self.verbose: bool   = verbose
        self.name            = name
        self.last_user_input: str = ''
        self.dalle   = Interface('dall_e')
        self.search  = Interface('serpapi')
        self.crawler = Interface('selenium')
        self.speech  = Interface('whisper')
        self.ocr     = Interface('ocr')

        self.short_term_memory: Memory = SlidingWindowListMemory(window_size=10)
        self.long_term_memory:  Memory = VectorDatabaseMemory(enabled=settings.SYMAI_CONFIG['INDEXING_ENGINE_API_KEY'] is not None, top_k=10)

        self._preprocessor  = ChatBot._init_custom_input_preprocessor(name=name, that=self)
        self._postprocessor = ChatBot._init_custom_input_postprocessor(that=self)

        self.detect_capability   = InContextClassification(SymbiaCapabilities())
        self.detect_memory_usage = InContextClassification(MemoryCapabilities())

        self.expression = Interface('wolframalpha')

    def repeat(self, query, **kwargs):
        return self.narrate('Symbia does not understand and asks to repeat and give more context.', prompt=query)

    def narrate(self, message: str, context: str = None, category: str = None, end: bool = False, **kwargs) -> Symbol:
        reflection = context if context is not None else ''
        ltmem_recall = 'No memories retrieved.'
        stmem_recall = '\n'.join(self.short_term_memory.recall())
        stmem_recall = stmem_recall if len(stmem_recall) > 0 else 'No memories retrieved.'

        if category == 'RECALL':
            ltmem_recall = '\n'.join(self.long_term_memory.recall(reflection))
            ltmem_recall = ltmem_recall if len(ltmem_recall) > 0 else 'No memories retrieved.'
            scratchpad   = self._memory_scratchpad(reflection, stmem_recall, ltmem_recall)
            memory_usage = str(self.detect_memory_usage(scratchpad))

            if self.verbose:
                logging.debug(f'Scratchpad:\n{scratchpad}\n')
                logging.debug(f'Memory usage:\n{memory_usage}\n')
                logging.debug(f'Retrieved from short-term memory:\n{stmem_recall}\n')
                logging.debug(f'Retrieved from long-term memory:\n{ltmem_recall}\n')

            do         = self._extract_category(memory_usage)
            reflection = self._extract_reflection(memory_usage)

            if do == 'SAVE':
                self.long_term_memory.store(f'{self.name}: {reflection}')
                if self.verbose: logging.debug(f'Store new long-term memory:\n{reflection}\n')
                message = f'{self.name} inform the user that the memory was stored.'

            elif do == 'DUPLICATE':
                message = f'{self.name} engages the user in a conversation about the duplicate topic, showing the user she remembered the past interaction.'

            elif do == 'IRRELEVANT':
                message = f'{self.name} discusses the topic with the user.'

            else: pass

        if self.verbose:
            logging.debug(f'Storing new short-term memory:\nUser: {self.last_user_input}\n')
            logging.debug(f'Storing new short-term memory:\n{self.name}: {reflection}\n')

        reply = f'{self.name}: {self._narration(message, self.last_user_input, reflection, ltmem_recall, stmem_recall, **kwargs)}'

        if end: print('\n\n', reply)

        return Symbol(reply)

    def input(self, message: str = "Please add more information", **kwargs) -> Symbol:
        @core.userinput(
            pre_processors=[self._preprocessor()],
            post_processors=[StripPostProcessor(),
                            self._postprocessor()],
            **kwargs
        )
        def _func(_, message) -> str:
            pass

        return Symbol(_func(self, message))

    @property
    def static_context(self) -> str:
        return ChatBot._symai_chat.format(self.name)

    @staticmethod
    def _init_custom_input_preprocessor(name, that):
        class CustomInputPreProcessor(ConsoleInputPreProcessor):
            def __call__(self, argument):
                msg     = re.sub(f'{name}:\s*', '', str(argument.args[0]))
                console = f'\n{name}: {msg}\n$> '

                if len(msg) > 0:
                    that.short_term_memory.store(f'{name}: ' + msg)

                return console

        return CustomInputPreProcessor

    @staticmethod
    def _init_custom_input_postprocessor(that):
        class CustomInputPostProcessor(ConsolePostProcessor):
            def __call__(self, rsp, argument):
                that.short_term_memory.store(f'User: {str(rsp)}')

                return rsp

        return CustomInputPostProcessor


    def _narration(self, msg: str, query: str, reflection: str, ltmem_recall: str, stmem_recall: str, **kwargs):
        prompt = f'''
{self._symai_chat.format(self.name)}

    [NARRATION](
{msg}
)

    [LONG-TERM MEMORY RECALL](
{ltmem_recall}
)

    [SHORT-TERM MEMORY RECALL](
{stmem_recall}
)

    [USER](
{query}
)

    [REFLECTION](
{reflection}
)

The chatbot always reply in the following format
{self.name}: <reply>
'''
        @core.zero_shot(prompt=prompt, **kwargs)
        def _func(_) -> str:
            pass
        if self.verbose: logging.debug(f'Narration:\n{prompt}\n')
        res = _func(self)
        res = res.replace(f'{self.name}: ', '').strip()
        return res

class SymbiaChat(ChatBot):
    def __init__(self, name: str = 'Symbia', verbose: bool = False, **kwargs):
        super().__init__(name=name, verbose=verbose, **kwargs)
        self.message = self.narrate(f'{self.name} introduces herself, writes a greeting message and asks how to help.', context=None)

    def forward(self, usr: Optional[str] = None) -> Symbol:
        loop = True
        ask_input = True
        if usr:
            ask_input = False
            usr = self._to_symbol(usr)

        # added step-by-step interaction with the user if input is provided
        while loop:
            # if no input is provided, ask for input
            if ask_input:
                usr = self.input(self.message)
            else:
                loop = False # break the loop after the first iteration

            self.last_user_input = usr
            if self.verbose: logging.debug(f'User:\n{usr}\n')

            if len(str(usr)) > 0:
                ctxt = str(self.detect_capability(usr))
            else:
                ctxt = '[DK]'

            if self.verbose: logging.debug(f'In-context:\n{ctxt}\n')

            if '[EXIT]' in ctxt:
                self.message = self.narrate(f'{self.name} writes friendly goodbye message.', context=None, end=True)
                break

            elif '[HELP]' in ctxt:
                reflection   = self._extract_reflection(ctxt)
                self.message = self.narrate(f'{self.name} ', context=reflection)

            elif '[RECALL]' in ctxt:
                reflection   = self._extract_reflection(ctxt)
                category     = self._extract_category(ctxt)
                self.message = self.narrate(f'{self.name} uses replies based on what has been recovered from the memory.', context=ctxt, category=category)

            elif '[DK]' in ctxt:
                reflection   = self._extract_reflection(ctxt)
                self.message = self.narrate(f'{self.name} is not sure about the message and references and asks the user for more context.', context=reflection)

            else:
                try:
                    if '[SYMBOLIC]' in ctxt:
                        q            = usr.extract("mathematical formula that WolframAlpha can solve")
                        rsp          = self.expression(q)
                        self.message = self.narrate(f'{self.name} replies to the user and provides the solution of the math problem.', context=rsp)

                    elif '[SEARCH]' in ctxt:
                        q            = usr.extract('user query request')
                        rsp          = self.search(q)
                        self.message = self.narrate(f'{self.name} replies to the user based on the online search results.', context=rsp)

                    elif '[CRAWLER]' in ctxt:
                        q    = usr.extract('URL from text')
                        q    = q.convert('proper URL, example: https://www.google.com')
                        site = self.crawler(q)
                        site.save('tmp.html')
                        self.message = self.narrate(f'{self.name} explains that the website is downloaded to the `tmp.html` file.')

                    elif '[SPEECH-TO-TEXT]' in ctxt:
                        q            = usr.extract('extract file path')
                        rsp          = self.speech(q)
                        self.message = self.narrate('Symbia replies to the user and transcribes the content of the audio file.', context=rsp)

                    elif '[TEXT-TO-IMAGE]' in ctxt:
                        q            = usr.extract('text for image creation')
                        rsp          = self.dalle(q)
                        self.message = self.narrate('Symbia replies to the user and provides the image URL.', context=rsp)

                    elif '[OCR]' in ctxt:
                        url          = usr.extract('extract url')
                        rsp          = self.ocr(url)
                        self.message = self.narrate('Symbia replies to the user and provides OCR text from the image.', context=rsp)

                    elif '[RETRIEVAL]' in ctxt:
                        file_path = usr.extract('extract file path')
                        file = self.open(file_path)
                        q    = usr.extract('user question')
                        rsp  = file.stream(
                            Sequence(
                                IncludeFilter('include only facts related to the user question: ' | q),
                                Outline()
                            )
                        )
                        self.message = self.narrate('Symbia replies to the user and outlines and relies to the user query.', context=rsp)

                    else:
                        q            = usr.extract('user query request')
                        rsp          = self.search(q)
                        reflection   = self._extract_reflection(ctxt)
                        self.message = self.narrate('Symbia tries to interpret the response, and if unclear asks the user to restate the statement or add more context.', context=reflection)

                except Exception as e:
                    reflection   = self._extract_reflection(ctxt)
                    self.message = self.narrate('Symbia apologizes and explains the user what went wrong.', context=str(e))

        return self.message

    def _extract_reflection(self, msg: str) -> str:
        res = re.findall(r'\(([^)]+)\)', msg)
        if len(res) > 0:
            return res.pop()
        return

    def _extract_category(self, msg: str) -> str:
        res = re.findall(r'\[([^]]+)\]', msg)
        if len(res) > 0:
            return res.pop()
        return

    def _memory_scratchpad(self, context, short_term_memory, long_term_memory):
        scratchpad = f'''
    [REFLECT](
Query:      {self.last_user_input}
Reflection: {self._extract_reflection(context)}
)

    [SHORT-TERM MEMORY RECALL](
{short_term_memory}
)

    [LONG-TERM MEMORY RECALL](
{long_term_memory}
)
'''

        return scratchpad

def run() -> None:
    with OpenAICostTracker() as tracker:
        chat = SymbiaChat()
        chat()
    print(tracker)

if __name__ == '__main__':
    run()

