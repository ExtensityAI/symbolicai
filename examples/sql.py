import symai as ai
from symai.post_processors import StripPostProcessor
from symai.pre_processors import PreProcessor
from symai.symbol import Expression, Symbol

SQL_CONTEXT = """[Description]
The following statements describe the Structured Query Language (SQL):

Commonly used SQL commands with examples:
Most SQL commands are used with operators to modify or reduce the scope of data operated on by the statement. Some commonly used SQL commands, along with examples of SQL statements using those commands, follow.

SQL SELECT. The SELECT command is used to get some or all data in a table. SELECT can be used with operators to narrow down the amount of data selected.
SQL CREATE. The CREATE command is used to create a new SQL database or SQL table. Most versions of SQL create a new database by creating a new directory, in which tables and other database objects are stored as files.
SQL DELETE. The DELETE command removes rows from a named table.
The CREATE TABLE command is used create a table in SQL.

[Examples]
// Select the title, author and publication date columns from a table named catalog.
SELECT title, author, pub_date
FROM catalog
WHERE pub_date = 2021;

// The following CREATE DATABASE statement creates a new SQL database named Human_Resources:
SQL: CREATE DATABASE Human_Resources;

// The following statement creates a table named Employees that has three columns: employee_ID, last_name and first_name, with the first column storing integer (int) data and the other columns storing variable character data of type varchar and a maximum of 255 characters.
SQL: CREATE TABLE Employees (
    employee_ID int,
    last_name varchar(255),
    first_name varchar(255)
);

// All records of employees with the last name Smithee are deleted:
SQL: DELETE FROM Employees WHERE last_name='Smithee';

[Last Example]
--------------
"""


class SQLPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args, **kwds):
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return '// {} SQL:'.format(str(wrp_self))


class SQL(Expression):
    @property
    def static_context(self):
        return SQL_CONTEXT

    def forward(self, sym: Symbol, *args, **kwargs):
        @ai.few_shot(prompt="Generate queries based on the SQL domain specific language description\n",
                     examples=[],
                     pre_processors=[SQLPreProcessor()],
                     post_processors=[StripPostProcessor()],
                     stop=[';'], **kwargs)
        def _func(_) -> str:
            pass
        return SQL(_func(SQL(sym)))

