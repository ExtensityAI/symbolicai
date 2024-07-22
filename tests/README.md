You can run the tests with the following command:

```bash
pytest
```

By default, it runs with the `v --ignore=tests/test_imports` flags.

We recommend to run the `test_imports.py` file separately, as it should be run only once (or occasionally, when you add new imports to the package):

```bash
pytest -q --tb=no tests/test_imports.py
```
