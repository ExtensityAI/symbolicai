from typing import List, Any, Optional, Dict, Callable
from .symbol import Symbol, Expression
from .components import Output
from symai import ai
import symai.backend.settings as settings


class ChatBot(Expression):
    _symai_chat: str = """This is a conversation between a chatbot ({}:) and a human (User:). It also includes narration Text (Narrator:) describing the next dialog.
    The chatbot primarily follows the narrative instructions, and then uses the user input to condition on for the generated response.
"""
    
    def __init__(self, value = None, name: str = 'Symbia', output: Optional[Output] = None, verbose: bool = False):
        super().__init__(value)
        that = self
        self.verbose: bool = verbose
        self.name = name
        self.last_user_input: str = ''
        self.memory: ai.Memory = ai.SlidingWindowListMemory()
        use_external_memory: bool = settings.SYMAI_CONFIG['INDEXING_ENGINE_API_KEY'] is not None
        self.long_term_memory: ai.Memory = ai.VectorDatabaseMemory(enabled=use_external_memory)
        Expression.command(engines=['symbolic'], expression_engine='wolframalpha')
        class CustomInputPreProcessor(ai.ConsoleInputPreProcessor):
            def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
                super().override_reserved_signature_keys(wrp_params, *args, **kwds)
                if f'{name}:' in Symbol(args[0]):
                    input_ = f'\n{str(args[0])}\n$> '
                else:
                    input_ = f'\n{name}: {str(args[0])}\n$> '
                that.memory.store(input_)
                return input_
        self._pre_processor = CustomInputPreProcessor
        def custom_post_processor(wrp_self, wrp_params, rsp, *args, **kwargs):
            that.memory.store(f'User: {str(rsp)}')
            return rsp
        self._post_processor = custom_post_processor
        
        self.init_options()
        Expression.command(engines=['symbolic'], expression_engine='wolframalpha')

    def repeat(self, query, **kwargs):
        return self.narrate('Symbia does not understand and asks to repeat and give more context.', prompt=query)
    
    def init_options(self):
        self.options_detection = ai.Classify(options=self.options)
        self.extended_options_detection = ai.Classify(options=self.extended_options)
    
    @property
    def options(self):
        return [
            'option 1 = [jokes, feelings, chit chat]',
            'option 2 = [task or command, query about facts, general help, open question, weather forecast, time, date, location, birth location, birth date, draw, speech, audio, ocr, open file, text to image, image to text, speech recognition, transcribe]',
            'option 3 = [exit, quit, bye, goodbye]',
            'option 4 = [chatbot features, list of commands, list of capabilities]',
            'option 5 = [follow up question, continuation, more information]',
            'option 6 = [mathematical equation, mathematical problem, mathematical question]',
            'option 9 = [none of the other, unknown, invalid, not understood]'
        ]
        
    @property
    def extended_options(self):
        return [
            'option 1 = [search, google, bing, yahoo, web, facts, location, weather, lookup, query, birthday, birth place, knowledge-based questions: what - where - who - why - how - which - whose]',
            'option 2 = [fetch, crawl, scrape, download http https, url dump]',
            'option 3 = [converse, small talk, ask about feeling, reply to a specific topic, chit chat, jokes, how are you, what colors do you like]',
            'option 4 = [wav, mp3, audio, speech, listen, transcribe, convert, convert audio to text]',
            'option 5 = [draw, create meme, generate image]',
            'option 6 = [scan image, read text from image, ocr, optical character recognition]',
            'option 7 = [open file, PDF, text file]',
            'option 9 = [none of the other, unknown, invalid, not understood]'
        ]
    
    @property
    def static_context(self) -> str:
        return ChatBot._symai_chat.format(self.name)
    
    def narrate(self, message: str, context: str = None, do_recall: bool = True, end: bool = False, **kwargs) -> "Symbol":
        narration = f'Narrator: {message}'
        self.memory.store(narration)
        ctxt = context if context is not None else ''
        value = f"{self.static_context}\n---------\nAdditional context and facts: {ctxt}\n---------\n" if ctxt is not None else ''
        value += '\nShort-term memory recall:\n'
        st_memory = self.memory.recall()
        value += '\n'.join(st_memory)
        value += '\nLong-term memory recall:\n'
        query = f'{self.last_user_input}'
        recall = self.long_term_memory.recall(query) if do_recall else []
        value += '\n'.join(recall)
        if self.verbose: print('[DEBUG] long-term memory recall: ', recall)
        value += f'\n{self.name}:'
        if self.verbose: print('[DEBUG] narration: ', narration)
        if self.verbose: print('[DEBUG] function context: ', ctxt)
        @ai.zero_shot(prompt=value, 
                      stop=['User:'], **kwargs)
        def _func(_) -> str:
            pass
        model_rsp = _func(self)
        rsp = f"{self.name}: {model_rsp}"
        if self.verbose: print('[DEBUG] model reply: ', model_rsp)
        memory = f"{self.last_user_input} >> {model_rsp}" if do_recall else ""
        if len(memory) > 0: self.long_term_memory.store(memory)
        if self.verbose: print('[DEBUG] store new memory reply: ', memory)
        sym = self._sym_return_type(rsp)
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
    
    @property
    def _sym_return_type(self):
        return ChatBot
    
    def forward(self, **kwargs):
        pass 


class SymbiaChat(ChatBot):
    @property
    def _sym_return_type(self):
        return SymbiaChat
    
    def forward(self):
        message = self.narrate('Symbia introduces herself, writes a greeting message and asks how to help.', do_recall=False)
        while True:
            # query user
            usr = self.input(message)
            self.last_user_input = usr
            if self.verbose: print('[DEBUG] user query: ', usr)
            
            # detect context
            if len(str(usr)) > 0:
                ctxt = str(self.options_detection(usr))
            else:
                ctxt = self.options[-1] # unknown default option

            if self.verbose: print('[DEBUG] options detected: ', ctxt)
            
            if 'exit' in ctxt or 'quit' in ctxt: # exit
                self.narrate('Symbia writes goodbye message.', end=True, do_recall=False)
                break # end chat
            
            elif 'option 4' in ctxt: # help
                message = self.narrate('Symbia writes for each capability one sentence.', 
                                       context=self.options, do_recall=False)
                      
            elif 'option 1' in ctxt: # chit chat
                message = self.narrate('Symbia replies to the user question in a casual way.')
                
            elif 'option 6' in ctxt: # solve a math problem
                q = usr.extract("mathematical formula that WolframAlpha can solve")
                rsp = q.expression()
                message = self.narrate('Symbia replies to the user and provides the solution of the math problem.', 
                                        context=rsp)
        
            elif 'option 2' in ctxt or 'option 5' in ctxt: 
                # detect command
                option = str(self.extended_options_detection(usr))
                if self.verbose: print('[DEBUG] extended options detected: ', option)
                
                try:
                    
                    if 'option 1' in option: # search request
                        q = usr.extract('user query request')
                        rsp = Expression.search(q)
                        message = self.narrate('Symbia replies to the user based on the online search results.', 
                                                context=rsp)                    
                    elif 'option 2' in option: # fetch a website
                        q = usr.extract('URL from text')
                        q = q.convert('proper URL, example: https://www.google.com')
                        site = Expression.fetch(q)
                        site.save('tmp.html')
                        message = self.narrate('Symbia explains that the website is downloaded to the `tmp.html` file.') 
                    
                    elif 'option 3' in option: # chatbot conversation
                        message = self.narrate('Symbia replies to the last user question.')
                        
                    elif 'option 4' in option: # speech to text
                        q = usr.extract('extract file path')
                        rsp = Expression.speech(q)
                        message = self.narrate('Symbia replies to the user and transcribes the content of the audio file.', 
                                                context=rsp)
                        
                    elif 'option 5' in option: # draw an image with DALL-E
                        q = usr.extract('text for image creation')
                        rsp = Expression.draw(q)
                        message = self.narrate('Symbia replies to the user and provides the image URL.', 
                                                context=rsp)
                        
                    elif 'option 6' in option: # perform ocr on an image
                        url = usr.extract('extract url')
                        rsp = ai.Expression.ocr(url)
                        message = self.narrate('Symbia replies to the user and provides OCR text from the image.', 
                                                context=rsp)
                        
                    elif 'option 7' in option: # scan a text-based document
                        file = usr.extract('extract file path')
                        q = usr.extract('user question')
                        expr = ai.Stream(
                            ai.Sequence(
                                ai.IncludeFilter('include only facts related to the user question: ' @ q),
                                ai.Outline()
                            )
                        )
                        rsp = expr(file)
                        message = self.narrate('Symbia replies to the user and outlines and relies to the user query.', 
                                                context=rsp)
                        
                    else: # failed or not implemented    
                        q = usr.extract('user query request')
                        rsp = Expression.search(q)                    
                        message = self.narrate('Symbia apologizes, tries to interpret the response and states that the capability is not available yet.', context=rsp, do_recall=False)
                        
                except Exception as e:
                    message = self.narrate('Symbia apologizes and explains the user what went wrong.',
                                            context=str(e))

            else: # repeat
                q = usr.extract('user query request')
                rsp = Expression.search(q)
                message = self.narrate('Symbia tries to interpret the response, and if unclear asks the user to restate the question or add more context.', context=rsp)


def run() -> None:
    chat = SymbiaChat()
    chat()
