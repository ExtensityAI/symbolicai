# Module Management

The `Import` class is a module management class in the SymbolicAI library. This class provides an easy and controlled way to manage the use of external modules in the user’s project, with main functions including the ability to install, uninstall, update, and check installed modules. It is used to manage expression loading from packages and accesses the respective metadata from the `package.json`.

The metadata for the package includes version, name, description, and expressions. It also lists the package dependencies required by the package.

Here is an example of a `package.json` file:

```json
{
    "version": "0.0.1",
    "name": "<username>/<repo_name>",
    "description": "<Project Description>",
    "expressions": [{"module": "src/func", "type": "MyExpression"}],
    "run": {"module": "src/func", "type": "MyExpression"},
    "dependencies": []
}
```

- `version`: Specifies the version number of the package. It is recommended to follow semantic versioning.
- `name`: Specifies the name of the package. It typically follows the format `<username>/<repo_name>`, where `<username>` is your GitHub username and `<repo_name>` is the name of your package repository.
- `description`: Provides a brief description of the package.
- `expressions`: Defines the exported expressions for the package. Each expression is defined by its `module` and `type`. The `module` specifies the file path or module name where the expression is defined, and the `type` specifies the type of the expression. These are used to be accessed from code by calling `Import.
- `run`: Specifies the expression that should be executed when the package is run. It follows the same format as the `expressions` property, only defined by a single entry point type.
- `dependencies`: Lists the package dependencies to other SymbolicAI packages! Dependencies can be specified with their package name `<username>/<repo_name>`.
## Working with Local Packages

The `Import` class supports both GitHub-hosted packages and local packages:

1. **Direct Local Imports**: You can directly import a package from a local directory by providing the full path:
   ```python
   local_module = Import("/path/to/local/module")
   ```

2. **Named Local Imports**: You can import a local package but register it with a specific name:
   ```python
   Import.install("username/repo_name", local_path="/path/to/local/module")
   ```

When working with local packages, the system will still look for a valid `package.json` file in the package root directory.

Note that the `package.json` file is automatically created when you use the Package Initializer tool (`symdev`) to create a new package. Alongside the `package.json` also a `requirements.txt` is created. This file contains all the `pip` relevant dependencies.

To import a package from code, see the following example:

```python
from symai import Import
symask_module = Import("ExtensityAI/symask")
```
You can also pass additional parameters to control installation behavior:

```python
# Install a module with Git submodules initialized
symask_module = Import("ExtensityAI/symask", submodules=True)

# Import from a local directory
local_module = Import("/path/to/local/module")

# Import using a local path but install as a named package
named_local = Import("username/repo_name", local_path="/path/to/local/module")
```

This command will clone the module from the given GitHub repository (`ExtensityAI/symask` in this case), install any dependencies, and expose the module's classes for use in your project.

You can also install a module without instantiating it using the `install` method:

```python
# Basic installation
Import.install("ExtensityAI/symask")

# Install with Git submodules
Import.install("ExtensityAI/symask", submodules=True)

# Install from a local directory
Import.install("username/repo_name", local_path="/path/to/local/module")
```

The `Import` class will automatically handle the cloning of the repository and the installation of dependencies that are declared in the `package.json` and `requirements.txt` files of the repository.
You can also update an installed package using the `update` method:

```python
# Basic update
Import.update("ExtensityAI/symask")

# Update including Git submodules
Import.update("ExtensityAI/symask", submodules=True)
```

Similarly, you can remove a package:

```python
Import.remove("ExtensityAI/symask")
```

Or list all installed packages:

```python
installed_packages = Import.list_installed()
```

Please refer to the comments in the code for more detailed explanations of how each method of the `Import` class works.