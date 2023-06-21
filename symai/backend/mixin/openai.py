SUPPORTED_MODELS = ['gpt-3.5-turbo', 'gpt-4', 'text-embedding-ada-002']


class OpenAIMixin:
    def api_pricing(self):
        if self.model == 'gpt-3.5-turbo':
            return {
                'input':  0.0015 / 1_000,
                'output': 0.0020 / 1_000
            }

        elif self.model == 'gpt-4':
            return {
                'input':  0.03 / 1_000,
                'output': 0.06 / 1_000
            }

        elif self.model == 'text-embedding-ada-002':
            return {
                'usage': 0.0001 / 1_000
            }

