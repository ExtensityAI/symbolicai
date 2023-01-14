from botdyn import *


HEADER_STYLE_DESCRIPTION = """Design a web app with HTML, CSS and inline JavaScript. 
Use dark theme and best practices for colors, text font, etc. 
Use Bootstrap for styling.
Do NOT remove the {{placeholder}} tag and do NOT add new tags into the body!"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>News</title>
    <script></script>
  </head>
  <body>
  <h1>News Headlines</h1>
  {{placeholder}}
  </body>
</html>"""

HTML_STREAM_STYLE_DESCRIPTION = """Style the elements according to the bootstrap library.
Replace the list items with a summary title and the item text. 
Add highlighting animations. 
Use best practices for colors, text font, etc."""


class News(Expression):
    def __init__(self, url: str, pattern: str, filters: List[Expression] = [], render: bool = False):
        super().__init__()
        self.url = url
        self.pattern = pattern
        self.render_ = render
        filters = filters if isinstance(filters, List) or isinstance(filters, tuple) else [filters]
        self.data_stream = Stream(Sequence(
            Clean(),
            Translate(),
            Outline(),
            *filters,
            Compose(f'Compose news paragraphs. Combine only facts that belong topic-wise together:\n'),
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
        
    def render(self, sym: Symbol) -> Symbol:
        tmp = self.header_style(self.html_template, max_tokens=2000)
        self.html_template_seq.template_ = str(tmp)
        res = '\n'.join([str(s) for s in self.html_stream(sym)])
        res = Symbol(str(tmp).replace('{{placeholder}}', res))
        return res
        
    def forward(self) -> Symbol:
        res = self.fetch(url=self.url, pattern=self.pattern)
        vals = []
        for news in self.data_stream(res):
            vals.append(str(news))
        res = Symbol(vals).cluster()
        sym = res.map()
        if not self.render_:
            return sym
        return self.render(sym)
