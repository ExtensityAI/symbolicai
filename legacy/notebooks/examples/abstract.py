from symai import *
from symai.components import *

HEADER_DESCRIPTION = """Design a web app with HTML, CSS and inline JavaScript.
Use dark theme and best practices for colors, text font, etc.
Use Bootstrap for styling.
Do NOT remove the {{placeholder}} tag and do NOT add new tags into the body!"""


class Abstract(Expression):

    def static_context(self):
        return HEADER_DESCRIPTION

    def __init__(self, path: str, filters: List[Expression] = [], **kwargs):
        super().__init__(**kwargs)
        self.path = path
        filters = filters if isinstance(filters, List) or isinstance(filters, tuple) else [filters]
        self.data_stream = Stream(Sequence(
            Clean(),
            Translate(),
            *filters,
            Compose(f'Write a paper summary. Keep all information with the corresponding citations:\n'),
        ))

    def forward(self, **kwargs) -> Symbol:
        res = self.open(self.path, **kwargs)
        data = ''
        template = self.header_style(self.html_template)
        paragraphs = []
        for section in self.data_stream(res):
            key = section.unique(paragraphs)
            paragraphs.append(str(key))
            self.html_template_seq.template_ = str(template)
            kwargs['dynamic_context'] = f'Context: Create a summary paragraph about {str(key)}\n'
            data += '\n'.join([str(s) for s in self.html_stream(section, **kwargs)])
        res = Symbol(str(template).replace('{{placeholder}}', data))
        return res
