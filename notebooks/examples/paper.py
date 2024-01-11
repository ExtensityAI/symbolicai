from symai import *
from symai.components import *


HEADER_STYLE_DESCRIPTION = """Design a web app with HTML, CSS and inline JavaScript.
Use dark theme and best practices for colors, text font, etc.
Use Bootstrap for styling.
Do NOT remove the {{placeholder}} tag and do NOT add new tags into the body!"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Paper Summary</title>
    <script></script>
  </head>
  <body>
  {{placeholder}}
  </body>
</html>"""

HTML_STREAM_STYLE_DESCRIPTION = """Style the elements according to the bootstrap library.
Replace the list items with a summary title and the item text.
Add highlighting animations.
Use best practices for colors, text font, etc."""


class Paper(Expression):
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
        self.header_style = Style(description=HEADER_STYLE_DESCRIPTION,
                                  libraries=['https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css',
                                             'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js',
                                             'https://ajax.googleapis.com/ajax/libs/jquery/3.6.1/jquery.min.js'])

        self.html_template = Symbol(HTML_TEMPLATE)
        self.html_template_seq = Template()
        self.html_stream = Stream(
            Sequence(
                self.html_template_seq,
                Style(description=HTML_STREAM_STYLE_DESCRIPTION)
            )
        )

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
