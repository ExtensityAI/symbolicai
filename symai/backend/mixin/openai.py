SUPPORTED_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-16k', 'text-embedding-ada-002']


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

        elif self.model == 'gpt-4':
            return {
                'input':  0.03 / 1_000,
                'output': 0.06 / 1_000
            }

        elif self.model == 'gpt-4-16k':
            return {
                'input':  0.06 / 1_000,
                'output': 0.12 / 1_000
            }

        elif self.model == 'gpt-4-1106-preview' or self.model == 'gpt-4-vision-preview':
            return {
                'input':  0.01 / 1_000,
                'output': 0.03 / 1_000
            }

        elif self.model == 'text-embedding-ada-002':
            return {
                'usage': 0.0001 / 1_000
            }

    def api_max_tokens(self):
        if self.model == 'gpt-3.5-turbo' or self.model == 'gpt-3.5-turbo-0613':
            return 4_096

        elif self.model == 'gpt-3.5-turbo-16k' or self.model == 'gpt-3.5-turbo-16k-0613':
            return 16_384

        elif self.model == 'gpt-4' or self.model == 'gpt-4-0613':
            return 8_192

        elif self.model == 'gpt-4-1106-preview' or self.model == 'gpt-4-vision-preview':
            return 128_000

        elif self.model == 'gpt-4-32k' or self.model == 'gpt-4-32k-0613':
            return 32_768

        elif self.model == 'text-davinci-003' or self.model == 'text-davinci-002':
            return 4_097

        elif self.model == 'text-curie-001' or \
             self.model == 'text-babbage-001' or \
             self.model == 'text-ada-001' or \
             self.model == 'davinci' or \
             self.model == 'curie' or \
             self.model == 'babbage' or \
             self.model == 'ada':
            return 2_049

        elif self.model == 'text-embedding-ada-002':
            return 8_191
