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
        
        self.capabilities = [
            'option 1 = [search, web, facts, location, weather, lookup, query, birthday, birth place]',
            'option 2 = [fetch, get, crawl, scrape, web]',
            'option 3 = [converse, talk, ask, answer, reply]',
            'option 4 = [audio, speech, listen',
            'option 5 = [draw, create meme, generate image]',
            #'option 6 = [scan, read image, ocr, file]',
            #'option 7 = [execute, run, code]',
            #'option 8 = [open file, PDF, text file]',
            'option 9 = [non of the other, unknown, invalid, not understood]'
        ]

        self.capabilities_choice = ai.Choice(cases=self.capabilities, # use static context instead
                                             default=self.capabilities[-1])
        
        self.detect_context = [
            'option 1 = [open question, jokes, how are you, chit chat]',
            'option 2 = [specific task or command, query about facts, weather forecast, time, date, location, birth location, birth date, draw, speech, audio]',
            'option 3 = [exit, quit, bye, goodbye]',
            'option 4 = [help, list of commands, list of capabilities]',
            'option 5 = [follow up question, continuation, more information]',
            'option 6 = [non of the other, unknown, invalid, not understood]'
        ]

        self.context_choice = ai.Choice(cases=self.detect_context, # use static context instead
                                        default=self.detect_context[-1])

    def repeat(self, query, **kwargs):
        return self.narrate('Symbia does not understand and asks to repeat and give more context.', prompt=query)
    
    @property
    def static_context(self) -> str:
        return ChatBot._symai_chat.format(self.name)
    
    def narrate(self, message: str, context: str = None, end: bool = False, verbose=False, **kwargs) -> "Symbol":
        narration = f'Narrator: {message}'
        self.history.append(narration)
        ctxt = context if context is not None else ''
        value = f"Additional context and facts: {ctxt}\n---------\n" if ctxt is not None else ''
        value += '\n'.join(self.history[-10:])
        value += f'\n{self.name}:'
        if verbose: print('[DEBUG] narrate: ', narration)
        if verbose: print('[DEBUG] context: ', ctxt)
        @ai.zero_shot(prompt=value, 
                      stop=['User:'], **kwargs)
        def _func(_) -> str:
            pass
        rsp = f"{self.name}: {_func(self)}"
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
    
    def forward(self, **kwargs):
        pass 


class SymbiaChat(ChatBot):
    def forward(self) -> str:
        message = self.narrate('Symbia introduces herself, writes a greeting message and asks how to help.')        
        while True:
            # query user
            usr = self.input(message)
            
            # detect context
            ctxt = self.context_choice(usr)
            
            if 'option 3' in ctxt: # exit
                self.narrate('Symbia writes goodbye message.', end=True)
                break # end chat
            
            elif 'option 4' in ctxt: # help
                message = self.narrate('Symbia writes for each capability one sentence.', 
                                       context=self.capabilities)
                      
            elif 'option 1' in ctxt: # chit chat
                message = self.narrate('Symbia replies to the user question in a casual way.')
        
            elif 'option 2' in ctxt: 
                # detect command
                option = self.capabilities_choice(usr)
                
                try:
                    
                    if 'option 1' in option:
                        q = usr.extract('user query request')
                        rsp = self.search(q)
                        message = self.narrate('Symbia replies to the user based on the online search results.', 
                                                context=rsp)                    
                    elif 'option 2' in option:
                        q = usr.extract('URL from text')
                        q = q.convert('proper URL, example: https://www.google.com')
                        site = self.fetch(q)
                        site.save('tmp.html')
                        message = self.narrate('Symbia explains that the website is downloaded to the `tmp.html` file.') 
                    
                    elif 'option 3' in option:
                        message = self.narrate('Symbia replies to the last user question.')
                        
                    elif 'option 4' in option:
                        q = usr.extract('extract file path')
                        rsp = self.speech(q)
                        message = self.narrate('Symbia replies to the user and transcribes the content of the audio file.', 
                                                context=rsp)
                        
                    elif 'option 5' in option:
                        q = usr.extract('text for image creation')
                        rsp = q.draw()
                        message = self.narrate('Symbia replies to the user and provides the image URL.', 
                                                context=rsp)
                        
                    else:
                        
                        message = self.narrate('Symbia apologizes and states that the capability is not available yet.')
                        
                except Exception as e:
                    
                    message = self.narrate('Symbia apologizes and explains the user what went wrong.',
                                            context=str(e))

            else: # repeat
                message = self.narrate('Symbia apologizes and asks the user to restate the question and add more context.')
