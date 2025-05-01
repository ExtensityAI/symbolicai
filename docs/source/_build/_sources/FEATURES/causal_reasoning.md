# 🧠 Causal Reasoning

The main goal of our framework is to enable reasoning capabilities on top of the statistical inference of Language Models (LMs). As a result, our `Symbol` objects offers operations to perform deductive reasoning expressions. One such operation involves defining rules that describe the causal relationship between symbols. The following example demonstrates how the `&` operator is overloaded to compute the logical implication of two symbols.

```python
from symai import Symbol

s1 = Symbol('The horn only sounds on Sundays.', only_nesy=True)
s2 = Symbol('I hear the horn.')

(s1 & s2).extract('answer') # Since I hear the horn, and the horn only sounds on Sundays, it must be Sunday.
```
> **Note**: The first symbol (e.g., `s1`) needs to have the `only_nesy` flag set to `True` for logical operators. This is because, without this flag, the logical operators default to string concatenation. While we didn't find a better way to handle meta-overloading in Python, this flag allows us to use operators like `'A' & 'B' & 'C'` to produce `'ABC'` or `'A' | 'B' | 'C'` to result in `'A B C'`. This syntactic sugar is essential for our use case.

The current `&` operation overloads the `and` logical operator and sends `few-shot` prompts to the neural computation engine for statement evaluation. However, we can define more sophisticated logical operators for `and`, `or`, and `xor` using formal proof statements. Additionally, the neural engines can parse data structures prior to expression evaluation. Users can also define custom operations for more complex and robust logical operations, including constraints to validate outcomes and ensure desired behavior.

To provide a more comprehensive understanding, we present several causal examples below. These examples aim to obtain logical answers based on questions like:

```python
# 1) "A line parallel to y = 4x + 6 passes through (5, 10). What is the y-coordinate of the point where this line crosses the y-axis?"
# 2) "Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother?"
# 3) "Is 1000 bigger than 1063.472?"
```

An example approach using our framework would involve identifying the neural engine best suited for the task and preparing the input for that engine. Here's how we could achieve this:

```python
from symai.extended import Interface

val = "<one of the examples above>"

# First, define a class that inherits from the Expression class
class ComplexExpression(ai.Expression):
    # write a method that returns the causal evaluation
    def causal_expression(self):
        pass # see below for implementation

# instantiate an object of the class
expr = ComplexExpression(val)
# set WolframAlpha as the main expression engine to use
wolfram = Interface('wolframalpha')
# evaluate the expression
res = expr.causal_expression()
```

A potential implementation of the `causal_expression` method could resemble the following:

```python
def causal_expression(self):
    # verify which case to use based on `self.value`
    if self.isinstanceof('mathematics'):
        # get the mathematical formula
        formula = self.extract('mathematical formula')
        # verify the problem type
        if formula.isinstanceof('linear function'):
            # prepare for WolframAlpha
            question = self.extract('question sentence')
            req = question.extract('what is requested?')
            x = self.extract('coordinate point (.,.)') # get the coordinate point / could also ask for other points
            query = formula | f', point x = {x}' | f', solve {req}' # concatenate the question and formula
            res = wolfram(query) # send the prepared query to WolframAlpha

        elif formula.isinstanceof('number comparison'):
            res = wolfram(formula) # send directly to WolframAlpha

        ... # more cases

    elif self.isinstanceof('linguistic problem'):
        sentences = self / '.' # first, split into sentences
        graph = {} # define the graph
        for s in sentences:
            sym = ai.Symbol(s)
            relations = sym.extract('connected entities (e.g., A has three B => A | A: three B)') / '|' # and split by pipe
            for r in relations:
                ... # add relations and populate the graph, or alternatively, learn about CycleGT

    ... # more cases
    return res
```

In the example above, the `causal_expression` method iteratively extracts information, enabling manual resolution or external solver usage.

> **[NOTE]**: Keep in mind that this implementation sketch requires significantly more engineering effort for the `causal_expression` method. Additionally, the current LLMs may sometimes struggle to extract accurate information or make correct comparisons. However, we believe that future advances in the field, specifically fine-tuned models like ChatGPT with Reinforcement Learning from Human Feedback (RLHF), will improve these capabilities.

Lastly, with sufficient data, we could fine-tune methods to extract information or build knowledge graphs using natural language. This advancement would allow the performance of more complex reasoning tasks, like those mentioned above. Therefore, we recommend exploring recent publications on [Text-to-Graphs](https://aclanthology.org/2020.webnlg-1.8.pdf). In this approach, answering the query involves simply traversing the graph and extracting the necessary information.
