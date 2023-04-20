import logging
import symai as ai
import torch
from functools import partial


SUB_TASKS_DESCRIPTION = """[Description]
You need to read the problem statement and break it down into smaller tasks.
Use tools like search engines (tool_search) or user feedback (tool_feedback) to get more information if something is unclear.
Use Q1, Q2, Q3, ... to represent questions and A1, A2, A3, ... to represent answers.
Use END to mark the end of a question or answer.

[Example]
For example, if the problem statement is to build a meme generator, you can break it down into the following sub-tasks:

[Problem Statement]
Build a meme generator.

[Sub-tasks]
Q1: tool_search: What is a meme? END
Q2: tool_search: Are there any meme generators examples already available? END
Q3: tool_feedback: What programming language should I use? END

[Answers]
A1: A meme is a humorous image, video, piece of text, etc., that is copied and spread rapidly by Internet users. END
A2: Yes, there are many meme generators available online. For example, https://imgflip.com/memegenerator. END
A3: Use Python. END
"""


class GraphCoherencePreProcessor(ai.PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args, **kwds):
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return '[Problem Statement]\n{}\n\n[Sub-tasks]\n'.format(str(wrp_self))


class GraphCoherence(ai.Expression):
    @property
    def _sym_return_type(self):
        return GraphCoherence

    @property
    def static_context(self) -> str:
        return SUB_TASKS_DESCRIPTION

    def forward(self, **kwargs):
        @ai.zero_shot(prompt="Break out a list of sub-tasks.",
                      pre_processor=[GraphCoherencePreProcessor()],
                      stop=['[Answers]'],
                      **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))


class GraphCoherenceChat(ai.ChatBot):

    @property
    def options(self):
        return [
            'option 1: [search request, find, lookup, query]',
            'option 2: [compile request, build, generate, create]',
            'option 3: [exit, stop program, quit, bye]',
            'option 4: [unknown, not sure, not recognized, not understood]'
        ]

    @ai.cache(in_memory=False)
    def embed_opt(self):
        opts = map(ai.Symbol, self.options)
        embeddings = [opt.embed() for opt in opts]

        return embeddings

    def classify(self, usr: ai.Symbol):
        '''
        Classify user input into one of the options.
        '''
        assert isinstance(usr, ai.Symbol)
        usr_embed = usr.embed()
        embeddings = self.embed_opt()
        similarities = [usr_embed.similarity(emb) for emb in embeddings]
        similarities = sorted(zip(self.options, similarities), key=lambda x: x[1], reverse=True)

        return similarities[0][0]

    def forward(self):
        logging.debug('Symbia is starting...')
        message = self.narrate('Symbia introduces herself, writes a greeting message and asks how to help.')
        while True:
            # query user
            usr = self.input(message)

            # TODO: update history context
            self.history.append(str(usr))
            logging.debug(f'History status: {self.history}')

            # TODO: detect context
            option = self.classify(usr)

            # TODO: this stuff
            # m_user = ai.Metric(usr, lambda_=lambda x, y: x.cosine_similarity(y))
            # res = m_user(embeddings, top=1) # if lambda_ is None, then use cosine similarity

            engines = ['neurosymbolic']

            # BUG: debug why log does not log
            # expr = ai.Log(ai.Trace(self.options_detection, engines=engines), engines=engines)
            # option = expr(usr)
            # logging.debug(f'Detected option: {option}')
            # gc_usr = GraphCoherence(usr)
            # expr = ai.Log(ai.Trace(gc_usr, engines=engines), engines=engines)
            # subtasks = expr()
            # logging.debug(subtasks)

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

                elif 'option 3' in option: # exit
                    self.narrate('Symbia writes goodbye message.', end=True)
                    break # end chat

                else: # failed or not implemented
                    message = self.narrate('Symbia apologizes and states that the capability is not available yet.')

            except Exception as e:

                message = self.narrate('Symbia apologizes and explains the user what went wrong.',
                                        context=str(e))


if __name__ == '__main__':
    logging.debug('Starting GraphCoherence')
    chat = GraphCoherenceChat()
    logging.debug('Initializing options')
    chat.init_options()
    logging.debug('Starting chat')
    chat.forward()
