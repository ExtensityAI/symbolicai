import logging
import re

from loguru import logger

from . import core
from .backend.settings import HOME_PATH
from .components import InContextClassification
from .interfaces import cfg_to_interface
from .memory import SlidingWindowListMemory
from .post_processors import ConsolePostProcessor, StripPostProcessor
from .pre_processors import ConsoleInputPreProcessor
from .prompts import MemoryCapabilities, SymbiaCapabilities
from .symbol import Expression, Symbol
from .utils import UserMessage

logging.getLogger('charset_normalizer').setLevel(logging.ERROR)


class ChatBot(Expression):
    _symai_chat: str = '''This is a conversation between a chatbot (Symbia:) and a human (User:). The chatbot follows a narrative structure, primarily relying on the provided instructions. It uses the user's input as a conditioning factor to generate its responses. Whenever Symbia retrieves any long-term memories, it checks the user's query and incorporates information from the long-term memory buffer into its response. If the long-term memories cannot provide a suitable answer, Symbia then checks its short-term memory to be aware of the topics discussed in the recent conversation rounds. Your primary task is to reply to the user's question or statement by generating a relevant and contextually appropriate response. Do not focus on filling the scratchpad with narration, long-term memory recall, short-term memory recall, or reflections. Always consider any follow-up questions or relevant information from the user to generate a response that is contextually relevant. Endeavor to reply to the greatest possible effort.'''

    def __init__(
        self,
        value: str | None = None,
        name: str = 'Symbia',
        verbose: bool = False,
        short_term_mem_window_size: int = 10,
        long_term_mem_top_k: int = 10,
        index_name: str = 'symbia_index',
        **kwargs
    ):
        super().__init__(value, **kwargs)
        self.sym_return_type = ChatBot
        self.verbose: bool = verbose
        self.name = name
        self.index_name = index_name
        self.interfaces = cfg_to_interface()
        self.short_term_memory = SlidingWindowListMemory(window_size=short_term_mem_window_size)
        self.long_term_mem_top_k = long_term_mem_top_k
        self.long_term_memory = self.interfaces['indexing']
        self._preprocessor = ChatBot._init_custom_input_preprocessor(name=name, that=self)
        self._postprocessor = ChatBot._init_custom_input_postprocessor(that=self)
        self.detect_capability = InContextClassification(SymbiaCapabilities())
        self.detect_memory_usage = InContextClassification(MemoryCapabilities())
        self._last_user_input: str = ''

    def repeat(self, query, **_kwargs):
        return self.narrate('Symbia does not understand and asks to repeat and give more context.', prompt=query)

    def narrate(self, message: str, context: str | None = None, category: str | None = None, end: bool = False, **kwargs) -> Symbol:
        reflection = context if context is not None else ''
        ltmem_recall = 'No memories retrieved.'
        stmem_recall = '\n'.join(self.short_term_memory.recall())
        stmem_recall = stmem_recall if len(stmem_recall) > 0 else 'No memories retrieved.'
        ltmem_recall = 'No memories retrieved.'

        if category == 'RECALL':
            if (HOME_PATH / 'localdb' / f'{self.index_name}.pkl').exists():
                ltmem_recall = '\n'.join(self.long_term_memory(reflection, operation='search', index_name=self.index_name))
            scratchpad = self._memory_scratchpad(reflection, stmem_recall, ltmem_recall)
            memory_usage = str(self.detect_memory_usage(scratchpad))

            if self.verbose:
                logger.debug(f'Scratchpad:\n{scratchpad}\n')
                logger.debug(f'Memory usage:\n{memory_usage}\n')
                logger.debug(f'Retrieved from short-term memory:\n{stmem_recall}\n')
                logger.debug(f'Retrieved from long-term memory:\n{ltmem_recall}\n')

            do = self._extract_category(memory_usage)
            reflection = self._extract_reflection(memory_usage)

            if do == 'SAVE':
                self.long_term_memory(f'{self.name}: {reflection}', operation='add', top_k=self.long_term_mem_top_k, index_name=self.index_name)
                self.long_term_memory('save', operation='config', index_name=self.index_name)
                if self.verbose:
                    logger.debug(f'Store new long-term memory:\n{reflection}\n')
                message = f'{self.name} inform the user that the memory was stored.'
            elif do == 'DUPLICATE':
                message = f'{self.name} engages the user in a conversation about the duplicate topic, showing the user she remembered the past interaction.'
            elif do == 'IRRELEVANT':
                message = f'{self.name} discusses the topic with the user.'

        if self.verbose:
            logger.debug(f'Storing new short-term memory:\nUser: {self._last_user_input}\n')
            logger.debug(f'Storing new short-term memory:\n{self.name}: {reflection}\n')

        reply = f'{self.name}: {self._narration(message, self._last_user_input, reflection, context, ltmem_recall, stmem_recall, **kwargs)}'

        if end:
            UserMessage(f'\n\n{reply}', text="extensity")

        return Symbol(reply)

    def input(self, message: str = "Please add more information", **kwargs) -> Symbol:
        @core.userinput(
            pre_processors=[self._preprocessor()],
            post_processors=[StripPostProcessor(), self._postprocessor()],
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
                msg = re.sub(f'{name}:\s*', '', str(argument.args[0]))
                console = f'\n{name}: {msg}\n$> '
                if len(msg) > 0:
                    that.short_term_memory.store(f'{name}: ' + msg)
                return console
        return CustomInputPreProcessor

    @staticmethod
    def _init_custom_input_postprocessor(that):
        class CustomInputPostProcessor(ConsolePostProcessor):
            def __call__(self, rsp, _argument):
                that.short_term_memory.store(f'User: {rsp!s}')
                return rsp
        return CustomInputPostProcessor

    def _narration(self, msg: str, query: str, reflection: str, context: str, ltmem_recall: str, stmem_recall: str, **kwargs):
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

    [CONTEXT](
{context}
)

The chatbot always reply in the following format
{self.name}: <reply>
'''
        @core.zero_shot(prompt=prompt, **kwargs)
        def _func(_) -> str:
            pass
        if self.verbose:
            logger.debug(f'Narration:\n{prompt}\n')
        return _func(self).replace(f'{self.name}: ', '').strip()

class SymbiaChat(ChatBot):
    def __init__(self, name: str = 'Symbia', verbose: bool = False, **kwargs):
        super().__init__(name=name, verbose=verbose, **kwargs)
        self.message = self.narrate(f'{self.name} introduces herself, writes a greeting message and asks how to help.', context=None)

    def forward(self, usr: str | None = None) -> Symbol:
        loop = True
        ask_input = True
        if usr:
            ask_input = False
            usr = self._to_symbol(usr)

        while loop:
            usr, loop = self._resolve_user_input(usr, loop, ask_input)
            self._last_user_input = usr
            self._log_verbose('User', usr)

            ctxt = self._context_from_user(usr)
            self._log_verbose('In-context', ctxt)

            if self._handle_exit_context(ctxt):
                break

            if self._handle_reflection_context(ctxt):
                continue

            self._handle_interface_context(usr, ctxt)

        return self.message

    def _resolve_user_input(self, usr: Symbol | None, loop: bool, ask_input: bool) -> tuple[Symbol, bool]:
        if ask_input:
            usr = self.input(self.message)
        else:
            loop = False
        return usr, loop

    def _log_verbose(self, title: str, content) -> None:
        if self.verbose:
            logger.debug(f'{title}:\n{content}\n')

    def _context_from_user(self, usr: Symbol) -> str:
        text = str(usr)
        if len(text) == 0:
            return '[DK]'
        return str(self.detect_capability(usr))

    def _handle_exit_context(self, ctxt: str) -> bool:
        if '[EXIT]' in ctxt:
            self.message = self.narrate(f'{self.name} writes friendly goodbye message.', context=None, end=True)
            return True
        return False

    def _handle_reflection_context(self, ctxt: str) -> bool:
        if '[HELP]' in ctxt:
            reflection = self._extract_reflection(ctxt)
            self.message = self.narrate(f'{self.name} ', context=reflection)
            return True
        if '[RECALL]' in ctxt:
            reflection = self._extract_reflection(ctxt)
            category = self._extract_category(ctxt)
            self.message = self.narrate(f'{self.name} uses replies based on what has been recovered from the memory.', context=ctxt, category=category)
            return True
        if '[DK]' in ctxt:
            reflection = self._extract_reflection(ctxt)
            self.message = self.narrate(f'{self.name} is not sure about the message and references and asks the user for more context.', context=reflection)
            return True
        return False

    def _handle_interface_context(self, usr: Symbol, ctxt: str) -> None:
        try:
            if '[SYMBOLIC]' in ctxt:
                q = usr.extract("mathematical formula that WolframAlpha can solve")
                rsp = self.interfaces['symbolic'](q)
                self.message = self.narrate(f'{self.name} replies to the user and provides the solution of the math problem.', context=rsp)
            elif '[SEARCH]' in ctxt:
                q = usr.extract('user query request')
                rsp = self.interfaces['search'](q)
                self.message = self.narrate(f'{self.name} replies to the user based on the online search results.', context=rsp)
            elif '[SCRAPER]' in ctxt:
                q = usr.extract('URL from text')
                q = q.convert('proper URL, example: https://www.google.com')
                rsp = self.interfaces['scraper'](q)
                self.message = self.narrate(f'{self.name} replies to the user and narrates its findings.', context=rsp)
            elif '[SPEECH-TO-TEXT]' in ctxt:
                q = usr.extract('extract file path')
                rsp = self.interfaces['stt'](q)
                self.message = self.narrate(f'{self.name} replies to the user and transcribes the content of the audio file.', context=rsp)
            elif '[TEXT-TO-IMAGE]' in ctxt:
                q = usr.extract('text for image creation')
                rsp = self.interfaces['drawing'](q)
                self.message = self.narrate('Symbia replies to the user and provides the image URL.', context=rsp)
            elif '[FILE]' in ctxt:
                file_path = usr.extract('extract file path')
                q = usr.extract('user question')
                rsp = self.interfaces['file'](file_path)
                self.message = self.narrate(f'{self.name} replies to the user and outlines and relies to the user query.', context=rsp)
            else:
                q = usr.extract('user query request')
                reflection = self._extract_reflection(ctxt)
                self.message = self.narrate(f'{self.name} tries to interpret the response, and if unclear asks the user to restate the statement or add more context.', context=reflection)

        except Exception as e:
            reflection = self._extract_reflection(ctxt)
            self.message = self.narrate(f'{self.name} apologizes and explains the user what went wrong.', context=str(e))

    def _extract_reflection(self, msg: str) -> str:
        res = re.findall(r'\(([^)]+)\)', msg)
        if len(res) > 0:
            return res.pop()
        return None

    def _extract_category(self, msg: str) -> str:
        res = re.findall(r'\[([^]]+)\]', msg)
        if len(res) > 0:
            return res.pop()
        return None

    def _memory_scratchpad(self, context, short_term_memory, long_term_memory):
        return f'''
    [REFLECT](
Query:      {self._last_user_input}
Reflection: {self._extract_reflection(context)}
)

    [SHORT-TERM MEMORY RECALL](
{short_term_memory}
)

    [LONG-TERM MEMORY RECALL](
{long_term_memory}
)
'''

def run() -> None:
    chat = SymbiaChat()
    chat()

if __name__ == '__main__':
    run()
