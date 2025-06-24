# Querying Data with SQL

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date information and give you a better picture of the current state of the library.

---

Click here for [interactive notebook](https://github.com/ExtensityAI/symbolicai/blob/main/notebooks/Queries.ipynb)

```python
import os
import warnings
warnings.filterwarnings('ignore')
print(os.getcwd())
from symai import *
from IPython.display import display
from examples.sql import SQL
sql = SQL()
```

The `SQL` expression is defined and passes as a global context the syntax of the SQL language. The SQL expression is then used to generate queries based on the given context. We can then use the generated queries to get data from a database.
The following query is sent to the neural computation engine and creates a query based on the given context:
```python
val = None
def _fun(x):
    global val
    val = x
Symbol._metadata.input_handler = _fun
```

```python
Symbol('I have this class').translate('German')
```

```python
q = sql('Select all users above the age of 30 from the Members table.');
display(q) # SELECT * FROM Members WHERE age > 30
```

We can now try to further manipulate the result by asking the model to incorporate additional information, such as filtering to a specific time range:

```python
res = q << 'and limit the query to the last 10 minutes'
display(res) # SELECT * FROM Members WHERE age > 30 AND timestamp > NOW() - INTERVAL 10 MINUTE
```

We can also try to remove unwanted fields from the generated query. Notice how the model tries to remove not only the given statement but attributes associated with them:

```python
res -= ' AND ...'
display(res) # SELECT * FROM Members WHERE age > 30
```

And we can now even convert our query to other familiar domain specific languages, such as `SQL` or `ActiveRecord`:

```python
sql_res = res.convert("ActiveRecord")
display(sql_res) # Members.where("age > ?", 30)
```

To wrap things up, we might want to go back full circle and ask the model to generate again the explanation based on the given query:

```python
answer_doc = res.query("What does this query do?")
display(answer_doc)
# The given SQL query retrieves all records from a table named `Members` where two conditions are met:
# 1. The `age` column of a member record must be greater than 30.
# 2. The `timestamp` column of a member record must be more recent than (or within)
# the last 10 minutes from the current time. More specifically: - `SELECT *` indicates that it will
# fetch all columns for the qualifying rows. - `FROM Members` specifies the table from which to fetch
#  the data. - `WHERE` introduces the conditions to filter which rows should be returned. - `age > 30`
#  filters the records to only include those where the age column value is greater than 30. - `AND`
# combines multiple conditions, meaning a record must meet all the specified conditions to be included.
# - `timestamp > NOW() - INTERVAL 10 MINUTE` checks for records with a timestamp within the last 10
# minutes. `NOW()` is a function that returns the current datetime. `INTERVAL 10 MINUTE` specifies a
# time interval of 10 minutes. By subtracting this interval from the current time, the query creates a
# datetime value that is 10 minutes before now. The condition `timestamp >` selects all records where
# the `timestamp` is more recent than that calculated time. This query would be useful for identifying
# active, older members (above 30 years old) who have engaged with the system or performed an action
# that updates their `timestamp` within the last 10 minutes.
```

Ask it in natural language to modify the query:

```python
answer = res.query("How can you limit the number of results to 30 for an SQL query?")
display(answer)
# To limit the number of results to 30 for an SQL query, you can use the `LIMIT` clause in your SQL
# statement. The `LIMIT` clause restricts the number of rows returned by the query. Here's how you modify
# your given SQL query to limit the results to 30: ```sql SELECT * FROM Members WHERE age > 30 LIMIT 30;```
# This query will retrieve a maximum of 30 records from the 'Members' table where the 'age' column value is greater than 30.
```

Even translate the explanation to a different language on the fly:

```python
locale = Symbol(answer_doc).translate('German')
display(locale)
```

Fixing the query on the fly if something goes wrong or the user quickly wants to adapt a query:

```python
sql.adapt(context="""Explanation: Never allow SELECT *, always use LIMIT to a max of x <= 50 entries, where x is the user specified limit.""")
```

```python
res = sql('Select all users above the age of 30 from the Members table.')
display(res) # SELECT * FROM Members WHERE age > 30 LIMIT 50
```

```python
sql.clear()
```

```python
res = sql('Select all users above the age of 30 from the Members table.')
display(res) # SELECT * FROM Members WHERE age > 30
```
