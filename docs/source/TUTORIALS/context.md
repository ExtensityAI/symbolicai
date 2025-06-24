# Handling Files and Long Context Lengths

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date information and give you a better picture of the current state of the library.

---

Click here for [interactive notebook](https://github.com/ExtensityAI/symbolicai/blob/main/notebooks/News.ipynb)

```python
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('../') # set the working directory to the root of the project
from symai import *
from symai.components import *
from IPython.display import display
```

We can create contextual prompts to define the semantic operations for our model. However, this takes away a lot of our context size and since the GPT-3 context length is limited to 4097 tokens, this might quickly become a problem. Luckily we can use the `Stream` processing expression. This expression opens up a data stream and computes the remaining context length for processing the input data. Then it chunks the sequence and computes the result for each chunk. The chunks can be processed with a `Sequence` expression, that allows multiple chained operations in a sequential manner.

In the following example we show how we can extract news from a particular website and try to recombine all individual chunks again by clustering the information among the chunks and then recombining them. This gives us a way to consolidate contextually related information and recombine them in a meaningful way. Furthermore, the clustered information can then be labeled by looking / streaming through the values within the cluster and collecting the most relevant labels.
<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img6.png" width="720px">
If we repeat this process, we now get a way of building up a hierarchical cluster with labels as entry points to allow information retrieval from our new data structure.
To make manners more concrete, lets take a look at how to build up a news generator. Let us first start by importing some pre-defined string constants. These constants are used to define how the text is processed and how we desire the output to be formatted.

## News Generator

```python
from examples.news import HEADER_STYLE_DESCRIPTION, HTML_TEMPLATE, HTML_STREAM_STYLE_DESCRIPTION
```

We sub-class the `Expression` class and define the `__init__` implementation. Similar to PyTorch we can define the graph in the `__init__` method and then call the `forward` method to compute the result. Afterwards, the data from the web URL is streamed through a `Sequence` of operations. This cleans the text from all the clutter such as `\n`, `\t`, etc. and then extracts the news from the text.
The news are then filtered and re-composed. The resulting news texts are then clustered and the clusters are labeled. The labeled clusters are then recombined to return a rendered HTML format.

```python
class News(Expression):
    """The `News` class sub-classes `Expression` and provides a way to fetch and render news from a given url. It uses a `Stream` object to process the news data, with a sequence of `Clean`, `Translate`, `Outline`, and `Compose` expressions.
    It also defines a `Style` for the header, and a `Symbol` for the HTML templates.
    """
    def __init__(self, url: str, pattern: str, filters: List[Expression] = [], render: bool = False, **kwargs):
        """The `News` class constructor requires three arguments - `url`, `pattern` and `filters`.
        * `url` is a `str` containing the url to fetch the news from.
        * `pattern` is a `str` containing the name of the search key to be found on the web page.
        * `filters` is a `List[Expression]` containing any additional filters that should be applied to the news data.
        It defaults to an empty list if not specified.
        The `News` class also has an optional `render` argument which is a `bool` indicating whether the news should be rendered. It defaults to `False` if not specified.
        """
        super().__init__(**kwargs)
        self.url = url
        self.pattern = pattern
        self.render_ = render
        self.crawler = Interface('selenium')
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
        """The `render` method takes a `Symbol` as an argument and returns a `Symbol` containing the rendered news.
        It first sets the `html_template_seq` property of the `html_stream` to the result of applying the `header_style` to the `html_template`.
        It then iterates through the `data_stream` and collects the strings resulting from each expression.
        These strings are combined into a single `Symbol` object which is then clustered.
        Finally, the `render` method applies the `html_template` to the clustered `Symbol` and returns the result.
        """
        tmp = self.header_style(self.html_template)
        self.html_template_seq.template_ = str(tmp)
        res = '\n'.join([str(s) for s in self.html_stream(sym)])
        res = Symbol(str(tmp).replace('{{placeholder}}', res))
        return res

    def forward(self) -> Symbol:
        """The `forward` method is used to fetch and process the news data.
        It first calls the `fetch` method with the `url` and `pattern` arguments.
        It then iterates through the `data_stream` and collects the `Symbol` object resulting from each expression.
        These `Symbol` objects are then combined into a single `Symbol` object which is then mapped.
        If `render` is `False`, the mapped `Symbol` is returned. Otherwise, the `render` method is called with the `Symbol` and the resulting `Symbol` is returned.
        """
        res = self.crawler(url=self.url, pattern=self.pattern)
        vals = []
        for news in self.data_stream(res):
            vals.append(str(news))
        res = Symbol(vals).cluster()
        sym = res.map()
        if not self.render_:
            return sym
        return self.render(sym)
```

Here you can try the news generator:
```python
# crawling the website and creating an own website based on its facts
news = News(url='https://www.cnbc.com/cybersecurity/',
            pattern='cnbc',
            filters=ExcludeFilter('sentences about subscriptions, licensing, newsletter'),
            render=True)
```

Since the generative process will evaluate multiple expression, we can use the `Trace` and `Log` classes to keep track of what is happening.

```python
expr = Log(Trace(news))
res = expr()
os.makedirs('results', exist_ok=True)
path = os.path.abspath('results/news.html')
res.save(path, replace=False)
```

## Streaming over a PDF File

Another example is to read in a PDF file and extract the text from it to create a website based on its content.

```python
from ..examples.paper import Paper
```

The process is fairly similar to the news generator. We first read in the PDF file and then stream the text through a sequence of operations. The text is then cleaned and the sentences are extracted. The sentences are then clustered and labeled. The labeled clusters are then recombined to return a rendered HTML format.

```python
paper = Paper(path='examples/paper.pdf')
expr = Log(Trace(paper))
res = expr(slice=(1, 1))
os.makedirs('results', exist_ok=True)
path = os.path.abspath('results/news.html')
res.save(path, replace=False)
```
