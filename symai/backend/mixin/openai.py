SUPPORTED_COMPLETION_MODELS = [
    'davinci-002',
]
SUPPORTED_CHAT_MODELS = [
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
    'gpt-4o-2024-11-20',
    'gpt-4o-mini',
    'chatgpt-4o-latest',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
]
SUPPORTED_REASONING_MODELS = [
    'o3-mini',
    'o4-mini',
    'o1',
    'o3'
]
SUPPORTED_EMBEDDING_MODELS = [
    'text-embedding-ada-002',
    'text-embedding-3-small',
    'text-embedding-3-large'
]


class OpenAIMixin:
    def api_max_context_tokens(self):
        if self.model == 'text-curie-001' or \
           self.model == 'text-babbage-001' or \
           self.model == 'text-ada-001' or \
           self.model == 'davinci' or \
           self.model == 'curie' or \
           self.model == 'babbage' or \
           self.model == 'ada':
               return 2_049
        if self.model == 'gpt-3.5-turbo' or \
           self.model == 'gpt-3.5-turbo-0613' or \
           self.model == 'gpt-3.5-turbo-1106':
               return 4_096
        if self.model == 'gpt-4' or \
           self.model == 'gpt-4-0613' or \
           self.model == 'text-embedding-ada-002' or \
           self.model == 'text-embedding-3-small' or \
           self.model == 'text-embedding-3-large':
               return 8_192
        if self.model == 'gpt-3.5-turbo-16k' or \
           self.model == 'gpt-3.5-turbo-16k-0613' or \
           self.model == 'davinci-002':
               return 16_384
        if self.model == 'gpt-4-32k' or \
           self.model == 'gpt-4-32k-0613':
               return 32_768
        if self.model == 'gpt-4-1106-preview' or \
           self.model == 'gpt-4-turbo-2024-04-09' or \
           self.model == 'gpt-4-turbo' or \
           self.model == 'gpt-4-1106' or \
           self.model == 'gpt-4o' or \
           self.model == 'gpt-4o-2024-11-20' or \
           self.model == 'gpt-4o-mini' or \
           self.model == 'chatgpt-4o-latest':
               return 128_000
        if self.model == 'o1' or \
           self.model == 'o3' or \
           self.model == 'o3-mini' or \
           self.model == 'o4-mini':
               return 200_000
        if self.model == 'gpt-4.1' or \
           self.model == 'gpt-4.1-mini' or \
           self.model == 'gpt-4.1-nano':
            return 1_047_576
        raise ValueError(f'Unsupported model: {self.model}')

    def api_max_response_tokens(self):
        if self.model == 'davinci-002':
            return 2_048
        if self.model == 'gpt-4-turbo' or \
           self.model == 'gpt-4-turbo-2024-04-09' or \
           self.model == 'gpt-4-1106-preview' or \
           self.model == 'gpt-3.5-turbo-1106' or \
           self.model == 'gpt-3.5-turbo-0613' or \
           self.model == 'gpt-3.5-turbo':
               return 4_096
        if self.model == 'gpt-4-0613' or \
           self.model == 'gpt-4':
               return 8_192
        if self.model == 'gpt-3.5-turbo-16k-0613' or \
           self.model == 'gpt-3.5-turbo-16k' or \
           self.model == 'gpt-4o-mini' or \
           self.model == 'gpt-4o' or \
           self.model == 'gpt-4o-2024-11-20' or \
           self.model == 'chatgpt-4o-latest':
               return 16_384
        if self.model == 'gpt-4.1' or \
           self.model == 'gpt-4.1-mini' or \
           self.model == 'gpt-4.1-nano':
            return 32_768
        if self.model == 'o1' or \
           self.model == 'o3' or \
           self.model == 'o3-mini' or \
           self.model == 'o4-mini':
               return 100_000
        raise ValueError(f'Unsupported model: {self.model}')

    def api_embedding_dims(self):
        if self.model == 'text-embedding-ada-002':
            return 1_536
        if self.model == 'text-embedding-3-small':
            return 1_536
        if self.model == 'text-embedding-3-large':
            return 3_072
        raise ValueError(f'Unsupported model: {self.model}')
