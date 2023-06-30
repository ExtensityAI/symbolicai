from ..components import Stream, Style, Template
from ..symbol import Expression, Symbol

HEADER_STYLE_DESCRIPTION = """[Description]
Design a web view of data with HTML.
Use dark theme and best practices for colors, text font, etc.
Use Bootstrap for styling.

[Examples]
Chose the appropriate HTML tags:
- Use URL links such as http://www.example.com to: <a href="www.example.com" ... />
- Use <img ...> tag for images: http://www.../image.jpeg <img ... />
- Use <h1> ... </h1> tags for the titles and headers: <p> ... </p> tags for paragraphs
- Use <ul> ... </ul> tags for unordered lists ['demo', 'example', ...]: <ul><li>demo</li><li>example</li>...</ul>

[Template]
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Generated HTML Results</title>
    <script></script>
  </head>
  <body>
    <div class="container">
        <h1>Rendered results:</h1>
        <!-- Only generate HTML tags based on to the USER_CONTEXT: -->
        {{placeholder}}
    </div>
  </body>
</html>"""

HTML_STREAM_STYLE_DESCRIPTION = """[Description]:
Style the elements according to the bootstrap library.
Add animations if appropriate.
DO NOT change the content values, only add styling.
DO NOT add <script/> tags!
USE ONLY CSS and HTML tags.
Use best practices for colors, text font, etc.
"""

HTML_TEMPLATE_STYLE = """
<!doctype html>
<html lang="en">
  <head>
    ... <!-- Imports of all Bootstrap CSS and JS libraries -->
  </head>
  <body>
    <div class="container">
        <h1>Rendered results:</h1>
        <!-- Placeholder for the styled user DATA: -->
        {{placeholder}}
    </div>
  </body>
</html>"""


class HtmlStyleTemplate(Expression):
    def __init__(self):
        super().__init__()
        self.html_template_seq = Template()
        self.html_template_seq.template_ = HEADER_STYLE_DESCRIPTION
        self.html_stream = Stream(
            self.html_template_seq
        )
        self.style_template = Style(description=HTML_STREAM_STYLE_DESCRIPTION,
                                       libraries=['https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css',
                                                  'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js',
                                                  'https://ajax.googleapis.com/ajax/libs/jquery/3.6.1/jquery.min.js'])

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        """The `render` method takes a `Symbol` as an argument and returns a `Symbol` containing the rendered news.
        It first sets the `html_template_seq` property of the `html_stream` to the result of applying the `header_style` to the `html_template`.
        It then iterates through the `data_stream` and collects the strings resulting from each expression.
        These strings are combined into a single `Symbol` object which is then clustered.
        Finally, the `render` method applies the `html_template` to the clustered `Symbol` and returns the result.
        """
        if type(sym) != Symbol:
            sym = Symbol(sym)
        html_data = list(self.html_stream(sym, **kwargs))
        style_data = [str(self.style_template(html,
                                              template=HTML_TEMPLATE_STYLE,
                                              placeholder='{{placeholder}}',
                                              **kwargs)) for html in html_data]
        res = '\n'.join(style_data)
        res = Symbol(res)
        return res
