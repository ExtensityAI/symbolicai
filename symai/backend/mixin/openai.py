SUPPORTED_MODELS = [
    'davinci-002',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-1106',
    'gpt-3.5-turbo-0613',
    'gpt-4',
    'gpt-4-0613',
    'gpt-4-1106-preview', # @NOTE: probabily obsolete; same price as 'gpt-4-turbo-2024-04-09' but no vision
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
    'gpt-4o',
    'text-embedding-ada-002',
    'text-embedding-3-small',
    'text-embedding-3-large'
]


class OpenAIMixin:
    def api_pricing(self):
        if self.model == 'gpt-3.5-turbo':
            return {
                'input':  0.0015 / 1_000,
                'output': 0.0020 / 1_000
            }

        elif self.model == 'gpt-3.5-turbo-16k':
            return {
                'input':  0.003 / 1_000,
                'output': 0.004 / 1_000
            }

        elif self.model == 'gpt-3.5-turbo-1106' or \
             self.model == 'gpt-3.5-turbo-0613':
            return {
                'input':  0.001 / 1_000,
                'output': 0.002 / 1_000
            }

        elif self.model == 'gpt-4' or \
             self.model == 'gpt-4-0613':
            return {
                'input':  0.03 / 1_000,
                'output': 0.06 / 1_000
            }

        elif self.model == 'gpt-4-1106-preview' or \
             self.model == 'gpt-4-turbo-2024-04-09' or \
             self.model == 'gpt-4-turbo':
            return {
                'input':  0.01 / 1_000,
                'output': 0.03 / 1_000
            }

        elif self.model == 'gpt-4o':
            return {
                'input':  0.005 / 1_000,
                'output': 0.015 / 1_000
            }

        elif self.model == 'text-embedding-ada-002':
            return {
                'usage': 0.0001 / 1_000
            }

        elif self.model == 'text-embedding-3-small':
            return {
                'usage': 0.00002 / 1_000
            }

        elif self.model == 'text-embedding-3-large':
            return {
                'usage': 0.00013 / 1_000
            }

    def api_max_context_tokens(self):
        if self.model == 'gpt-3.5-turbo' or \
           self.model == 'gpt-3.5-turbo-0613' or \
           self.model == 'gpt-3.5-turbo-1106':
            return 4_096

        elif self.model == 'gpt-3.5-turbo-16k' or \
             self.model == 'gpt-3.5-turbo-16k-0613' or \
             self.model == 'davinci-002':
            return 16_384

        elif self.model == 'gpt-4' or \
             self.model == 'gpt-4-0613':
            return 8_192

        elif self.model == 'gpt-4-1106-preview' or \
             self.model == 'gpt-4-turbo-2024-04-09' or \
             self.model == 'gpt-4-turbo' or \
             self.model == 'gpt-4-1106' or \
             self.model == 'gpt-4o':
            return 128_000

        elif self.model == 'gpt-4-32k' or \
             self.model == 'gpt-4-32k-0613':
            return 32_768

        elif self.model == 'text-curie-001' or \
             self.model == 'text-babbage-001' or \
             self.model == 'text-ada-001' or \
             self.model == 'davinci' or \
             self.model == 'curie' or \
             self.model == 'babbage' or \
             self.model == 'ada':
            return 2_049

        elif self.model == 'text-embedding-ada-002' or \
             self.model == 'text-embedding-3-small' or \
             self.model == 'text-embedding-3-large':
            return 8_191

        else:
            # default to similar as in gpt-3.5-turbo
            print(f'WARNING: Model <{self.model}> not supported, defaulting to 4.096 tokens. May result in unexpected behavior.')
            return 4_096

    def api_max_response_tokens(self):
        if self.model == 'davinci-002':
            return 2_048

        elif self.model == 'gpt-4o' or \
           self.model == 'gpt-4-turbo' or \
           self.model == 'gpt-4-turbo-2024-04-09' or \
           self.model == 'gpt-4-1106-preview' or \
           self.model == 'gpt-3.5-turbo-1106' or \
           self.model == 'gpt-3.5-turbo-0613' or \
           self.model == 'gpt-3.5-turbo':
            return 4_096

        elif self.model == 'gpt-4-0613' or \
             self.model == 'gpt-4':
            return 8_192

        elif self.model == 'gpt-3.5-turbo-16k-0613' or \
             self.model == 'gpt-3.5-turbo-16k':
            return 16_384

    def api_embedding_dims(self):
        if self.model == 'text-embedding-ada-002':
            return 1_536
        elif self.model == 'text-embedding-3-small':
            return 1_536
        elif self.model == 'text-embedding-3-large':
            return 3_072
        else:
            print(f'WARNING: Model <{self.model}> not supported, defaulting to 768 dims. May result in unexpected behavior.')
            return 768
