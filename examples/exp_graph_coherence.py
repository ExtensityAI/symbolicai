import logging

import symai as ai
import torch


class GraphCoherence(ai.ChatBot):
    @property
    def _sym_return_type(self):
        return GraphCoherence

    @property
    def options(self):
        return [
            'option 1 = [search request, find, lookup, query]',
            'option 2 = [compile request, build, generate, create]',
            'option 9 = [exit, stop program, quit, bye]'
    ]

    @ai.cache()
    def embed_opt(self):
        opts = map(ai.Symbol, self.options)
        embeddings = [opt.embed().value for opt in opts]

        return embeddings

    def forward(self) -> str:
        logging.debug('Symbia is starting...')
        message = self.narrate('Symbia introduces herself, writes a greeting message and asks how to help.')
        while True:
            # query user
            usr = self.input(message)

            # TODO: update history context
            self.history.append(str(usr))
            logging.debug(f'History status: {self.history}')

            # TODO: detect context
            embeddings = self.embed_opt()
            option = self.options_detection(usr)
            logging.debug(f'Detected option: {option}')

            try:
                # TODO: add more context options

                if 'option 1' in option: # search request
                    q = usr.extract('user query request')
                    rsp = self.search(q)
                    message = self.narrate('Symbia replies to the user based on the online search results.',
                                            context=rsp)

                elif 'option 2' in option: # compile request
                    pass # TODO: try to compile a solution

                    # TODO: update memory [cache]

                elif 'option 9' in option: # exit
                    self.narrate('Symbia writes goodbye message.', end=True)
                    break # end chat

                else: # failed or not implemented
                    message = self.narrate('Symbia apologizes and states that the capability is not available yet.')

            except Exception as e:

                message = self.narrate('Symbia apologizes and explains the user what went wrong.',
                                        context=str(e))


if __name__ == '__main__':
    logging.debug('Starting GraphCoherence')
    chat = GraphCoherence()
    logging.debug('Initializing options')
    chat.init_options()
    logging.debug('Starting chat')
    chat.forward()
