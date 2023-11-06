from symai import *

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
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
  </head>
  <body>
  <h1>News Headlines</h1>
  {{placeholder}}

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4" crossorigin="anonymous"></script>
  </body>
</html>"""

HTML_STREAM_STYLE_DESCRIPTION = f"""Style the elements according to the bootstrap library.
Replace the list items with a summary title and the item text.
Add highlighting animations.
Use best practices for colors, text font, etc.
Assume the elements are inside the `placeholder` tag of the following HTML template:
{HTML_TEMPLATE}"""

class News(Expression):
    """The `News` class sub-classes `Expression` and provides a way to fetch and render news from a given url. It uses a `Stream` object to process the news data, with a sequence of `Clean`, `Translate`, `Outline`, and `Compose` expressions.
    It also defines a `Style` for the header, and a `Symbol` for the HTML templates.
    """
    def __init__(self, url: str, pattern: str, filters: List[Expression] = [], render: bool = False):
        """The `News` class constructor requires three arguments - `url`, `pattern` and `filters`.
        * `url` is a `str` containing the url to fetch the news from.
        * `pattern` is a `str` containing the regex pattern to be used to match the news.
        * `filters` is a `List[Expression]` containing any additional filters that should be applied to the news data.
        It defaults to an empty list if not specified.
        The `News` class also has an optional `render` argument which is a `bool` indicating whether the news should be rendered. It defaults to `False` if not specified.
        """
        super().__init__()
        self.url = url
        self.fetch = Interface('selenium')
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

        self.html_stream = Stream(
            Sequence(
                Template(template=HTML_TEMPLATE),
                Style(description=HTML_STREAM_STYLE_DESCRIPTION,
                      libraries=['https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css',
                                 'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js',
                                 'https://ajax.googleapis.com/ajax/libs/jquery/3.6.1/jquery.min.js']),
            )
        )

    def render(self, sym: Symbol, **kwargs) -> Symbol:
        """The `render` method takes a `Symbol` as an argument and returns a `Symbol` containing the rendered news.
        It first sets the `html_template_seq` property of the `html_stream` to the result of applying the `header_style` to the `html_template`.
        It then iterates through the `data_stream` and collects the strings resulting from each expression.
        These strings are combined into a single `Symbol` object which is then clustered.
        Finally, the `render` method applies the `html_template` to the clustered `Symbol` and returns the result.
        """
        res = '\n'.join([str(s) for s in self.html_stream(sym, **kwargs)])
        res = Symbol(str(HTML_TEMPLATE).replace('{{placeholder}}', res))
        return res

    def forward(self, **kwargs) -> Symbol:
        """The `forward` method is used to fetch and process the news data.
        It first calls the `fetch` method with the `url` and `pattern` arguments.
        It then iterates through the `data_stream` and collects the `Symbol` object resulting from each expression.
        These `Symbol` objects are then combined into a single `Symbol` object which is then mapped.
        If `render` is `False`, the mapped `Symbol` is returned. Otherwise, the `render` method is called with the `Symbol` and the resulting `Symbol` is returned.
        """
        res = self.fetch(url=self.url, pattern=self.pattern)
        vals = []
        for news in self.data_stream(res, **kwargs):
            vals.append(str(news))
        res = Symbol(vals).cluster()
        sym = res.map()
        if not self.render_:
            return sym
        return self.render(sym, **kwargs)
