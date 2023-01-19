from symai.chat import ChatBot


class SymbiaChat(ChatBot):
    def format(self, text: str) -> str:
        message = self.narrate('Symbia introduces herself, writes a greeting message and asks how to help.')        
        while True:
            # query user
            res = self.input(message)
            
            # detect context
            ctxt = self.context_choice(res, attach=self.detect_context_template)
            
            if 'option 3' in ctxt: # exit
                self.narrate('Symbia writes goodbye message.', end=True)
                break # end chat
            
            elif 'option 4' in ctxt: # help
                message = self.narrate('Symbia writes for each capability one sentence.', 
                                       prompt=self.capabilities)
                      
            elif 'option 1' in ctxt: # chit chat
                message = self.narrate('Symbia replies to the user question in a chit chat style.')
        
            elif 'option 2' in ctxt: 
                # detect command
                res = self.capabilities_choice(res, attach=self.capabilities_template)
                
                if 'option 1' in res:
                    q = self.search(res)
                    message = self.narrate('Symbia replies to the user based on the online search results.', 
                                           prompt=q)
                elif 'option 2' in res:
                    pass
                
                # TODO: ... Still in development
                
            else: # repeat
                message = self.narrate('Symbia apologizes and asks the user to restate the question and add more context.')