import argparse
try:
    import z3
except ImportError:
    z3 = None

from .conversation import Conversation
from .. import core
from ..components import Execute
from ..post_processors import StripPostProcessor, CodeExtractPostProcessor
from ..pre_processors import PreProcessor
from ..prompts import Prompt
from ..symbol import Expression, Symbol

#############################################################################################
#
# Code for detecting different patterns in the user query
#
#############################################################################################


PROBLEM_CATEGORY_CONTEXT = """[Description]
Classify the query in categories related to mathematical problems.
The possible categories are listed in the [Classes] section.
Do not attempt to solve the problem, just classify it.
The $> symbol is the prompt for the user input.
The // is the predicted category, ending with EOF.

[Classes]
- Arithmetic formula
- Equations with one variable
- Implication and logical expressions
- Probability and statistics
- Linear algebra
- Linguistic problem with relations
- Unknown category

[Examples]
Here is a list of examples categorizing the problem statements:
$> Evaluate the logical implication: (True and False) or (True and True)
// Implication and logical expressions EOF
$> 2 + 2 * 2
// Arithmetic formula EOF
$> Solve the equation: 2x + 3 = 5
// Equations with one variable EOF
$> A line passes through the points (1, 2) and (3, 4). What is the slope of the line?
// Linear algebra EOF
$> A bag contains 3 red balls and 4 blue balls. What is the probability of drawing a red ball?
// Probability and statistics EOF
$> Max is 2 years older than his brother. In 5 years, Max will be 3 times as old as his brother. How old is Max now?
// Linguistic problem with relations EOF


[Last Example]
--------------
"""

class ProblemClassifierPreProcessor(PreProcessor):
    def __call__(self, argument):
        return '$> {}\n//'.format(str(argument.prop.instance))


class OptionsPreProcessor(PreProcessor):
    def __call__(self, argument):
        return '$> :{}: == :{}: =>'.format(str(argument.prop.instance), str(argument.args[0]))


class ProblemClassifier(Expression):
    @property
    def static_context(self):
        return PROBLEM_CATEGORY_CONTEXT

    def __eq__(self, other, **kwargs) -> bool:
        @core.few_shot(prompt="Verify equality of the following categories. Ignore typos, upper / lower case or singular / plural differences:\n",
                     examples=Prompt([
                         '$> :Arithmetic formula: == :Arithmetics formula: =>True EOF',
                         '$> :arithmetic formula: == :Arithmetic formula: =>True EOF',
                         '$> :arithmetic formula: == :arithmeticformula: =>True EOF',
                         '$> :arithmetic formula: == :Implication and logical expressions: =>False EOF',
                         '$> :Linear algebra: == :Implication and logical expressions: =>False EOF',
                         '$> :Linear algebra: == :Unknown category: =>False EOF',
                         '$> :Linear algebra: == :Linear algebra: =>True EOF',
                         '$> :Probability and statistics: == :Probabilities and statistics: =>True EOF',
                         '$> :PROBABILITY AND STATISTICS: == :Probability and statistics: =>True EOF',
                         '$> :PROBABILITY AND STATISTICS: == :UNKNOWN CATEGORY: =>False EOF',
                     ]),
                     pre_processors=[OptionsPreProcessor()],
                     post_processors=[StripPostProcessor()],
                     stop=['EOF'], **kwargs)
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def forward(self, **kwargs) -> str:
        @core.few_shot(prompt="Classify the user query to the mathematical classes:\n",
                     examples=[],
                     pre_processors=[ProblemClassifierPreProcessor()],
                     post_processors=[StripPostProcessor()],
                     stop=['EOF'], **kwargs)
        def _func(_) -> str:
            pass

        return ProblemClassifier(_func(self))


class FormulaCheckerPreProcessor(PreProcessor):
    def __call__(self, argument):
        return '$> {} =>'.format(str(argument.prop.instance))


class FormulaChecker(Expression):
    def forward(self, **kwargs) -> bool:
        @core.few_shot(prompt="Is the following statement in an explicit formula form without natural language text?:\n",
                     examples=Prompt([
                         '$> 2 + 2 * 2 =>True EOF',
                         '$> x + 2 = 3 =>True EOF',
                         '$> Set of all natural numbers =>False EOF',
                         '$> Probability of drawing a red ball =>False EOF',
                         '$> (a + b) * (a - b) =>True EOF',
                         '$> Add the square root of nine to the square root of x =>False EOF',
                         '$> Five plus two equals seven =>False EOF',
                         '$> 5 + 2 = 7 =>True EOF',
                         '$> x is seven =>False EOF',
                         '$> x = 7 =>True EOF',
                         '$> Anna has two apples. She gives one to her brother. How many apples does Anna have now? =>False EOF',
                         '$> 0.447662 =>True EOF',
                         '$> Subtract the x from y squared =>False EOF',
                         '$> The sum of the first n natural numbers =>False EOF',
                         '$> Sum[x=5, {i=0, n=10}] =>True EOF',
                     ]),
                     pre_processors=[FormulaCheckerPreProcessor()],
                     post_processors=[StripPostProcessor()],
                     stop=['EOF'], **kwargs)
        def _func(_) -> bool:
            pass
        return _func(self)


#############################################################################################
#
# Code for rewriting user queries
#
#############################################################################################


class FormulaWriterPreProcessor(PreProcessor):
    def __call__(self, argument):
        return '$> {} =>'.format(str(argument.prop.instance))


class FormulaWriter(Expression):
    def forward(self, **kwargs) -> str:
        @core.few_shot(prompt="Rewrite the following natural language statement in a mathematical formula or higher-order logic statement to be solved by Mathematica:\n",
                     examples=Prompt([
                         '$> Add 5 plus 3 =>5 + 3 EOF',
                         '$> Seventy plus twenty =>70 + 20 EOF',
                         '$> Divide 5 by three =>5 / 3 EOF',
                         '$> The square root of pi plus x. =>Sqrt[Pi + x] EOF',
                         '$> Eight point five six seven one four two seven =>8.5671427 EOF',
                         '$> Give a solution for a quadratic equation x^2 + 2x + 1 =>Solve[x^2 + 2x + 1 ==0, x] EOF',
                         '$> Sum x n times from i equals 0 to n equals 10. x is equals to 5. =>Sum[x=5, {i=0, n=10}] EOF',
                         '$> Multiply the first statement in brackets a plus b times the second term in brackets c minus d =>(a + b) * (c - d) EOF'
                     ]),
                     pre_processors=[FormulaWriterPreProcessor()],
                     post_processors=[StripPostProcessor()],
                     stop=['EOF'], **kwargs)
        def _func(_) -> str:
            pass
        return _func(self)


LOGIC_TEMPLATE = """
# imports the available functions from the z3 library
from z3 import Solver, Function, IntSort, EnumSort, Int, And, Or, Xor, Const

# Define the problem statement as a function that takes a solver as input and returns a query constant as output
def problem_statement(S: Solver) -> Const:
    # Example for using the solver:
    # Porp, (A, B, C) = EnumSort('Prop', ('A', 'B', 'C')) # Define an enumerated sort
    # p = Function('prop_func', IntSort(), Prop)          # Define an uninterpreted function that takes an integer as input and returns a Prop as output
    # S.add(B == p(2))                                    # Assert a new fact
    # S.add(And(p(1) == A, p(2) == B, p(3) == C))         # Assert a new fact
    # ...                                                 # Define more facts
    # query = Const("query", Prop)                        # Create a new constant
    #
    # TODO: Define the logic expressions here using the S variable as the solver.
    query = None
    # insert your code here
    return query

# assign result to global output variable to make accessible for caller
_value_obj_ = problem_statement
"""


class SATSolver(Expression):
    def forward(self, code):
        assert z3 is not None, "The z3 library is not installed. Please install it using `pip install 'symbolicai[solver]'` and try again."
        # Create the execution template
        runner    = Execute(enclosure=True)
        # Execute the code
        statement = runner(code)
        # Create a new solver instance
        S         = z3.Solver()
        # Create a new query
        query     = statement['locals']['_output_'](S)
        # Check if the query can be solved
        r         = S.check()
        # Print the solution
        if r == z3.sat:
            # Get the model
            m = S.model()
            # Return the solution
            return m[query]
        else:
            print("Cannot solve the puzzle. Returned: " + str(r))
            return None


#############################################################################################
#
# Code for switching between different mathematical modules
#
#############################################################################################


class Solver(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sym_return_type = Solver
        self.solver = SATSolver()
        self.conv   = Conversation(init=LOGIC_TEMPLATE)
        self.pp     = CodeExtractPostProcessor()

    def rewrite_formula(self, sym, **kwargs):
        formula = sym
        check = FormulaChecker(formula)
        if not check(**kwargs):
            formula = FormulaWriter(sym)
            formula = formula(**kwargs)
        return formula

    def forward(self, sym: Symbol, **kwargs) -> str:
        classifier = ProblemClassifier(sym)
        problem = classifier(**kwargs)

        if 'Arithmetics formula' == problem:
            formula = self.rewrite_formula(sym, **kwargs)
            print(formula)
        elif 'Equations' == problem:
            formula = self.rewrite_formula(sym, **kwargs)
            print(formula)
        elif 'Implication and logical expressions' == problem:
            res     = self.conv(sym, **kwargs)
            code    = self.pp(str(res), None, tag="python")
            formula = self.solver(code, lambda: 'German')
            print(formula)
        elif 'Probability and statistics' == problem:
            raise NotImplementedError('This feature is not yet implemented.')
        elif 'Linear algebra' == problem:
            raise NotImplementedError('This feature is not yet implemented.')
        elif 'Linguistic problem with relations' == problem:
            raise NotImplementedError('This feature is not yet implemented.')
        else:
            return "Sorry, something went wrong. Please check if your backend is available and try again or report an issue to the devs. :("


def process_query(args) -> None:
    query = args.query
    solver = Solver()
    res = solver(query)
    print(res)


def run() -> None:
    # All the logic of argparse goes in this function
    parser = argparse.ArgumentParser(description='Welcome to the Symbolic<AI/> Shell support tool!')
    parser.add_argument('query', type=str, help='The prompt for the shell query.')

    args = parser.parse_args()
    process_query(args)


if __name__ == "__main__":
    run()
