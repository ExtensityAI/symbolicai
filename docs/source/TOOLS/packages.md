# Packages

### 📦 Package Manager

We provide a package manager called `sympkg` that allows you to manage extensions from the command line. With `sympkg`, you can install, remove, list installed packages, or update a module.

To use `sympkg`, follow the steps below:

1. Open your terminal or PowerShell.
2. Run the following command: `sympkg <command> [<args>]`

The available commands are:

- `i` or `install`: Install a new package. To install a package, use the following command: `sympkg i <package>`
- `r` or `remove`: Remove an installed package. To remove a package, use the following command: `sympkg r <package>`
- `l` or `list`: List all installed packages. To list installed packages, use the following command: `sympkg l`
- `u` or `update`: Update an installed package. To update a package, use the following command: `sympkg u <package>`

For more information on each command, you can use the `--help` flag. For example, to get help on the `i` command, use the following command: `sympkg i --help`.

Note: The package manager is based on GitHub, so you will need `git` installed to install or update packages. The packages names use the GitHub `<username>/<repo_name>` convention.

Happy package managing!

### 📦 Package Runner

The Package Runner is a command-line tool that allows you to run packages via alias names. It provides a convenient way to execute commands or functions defined in packages. You can access the Package Runner by using the `symrun` command in your terminal or PowerShell.

#### Usage

To use the Package Runner, you can run the following command:

```bash
$> symrun <alias> [<args>] | <command> <alias> [<package>]
```

The most commonly used Package Runner commands are:

- `<alias> [<args>]`: Run an alias
- `c <alias> <package>`: Create a new alias
- `l`: List all aliases
- `r <alias>`: Remove an alias

#### Examples

Here are a few examples to illustrate how to use the Package Runner:

The following command runs the specified `my_alias` with the provided arguments `arg1`, `arg2`, `kwarg1` and `kwarg2`, where `arg1` and `arg2` are considered as *args parameter and `kwarg1` and `kwarg1` **kwargs key-value arguments. These arguments will be passed on to the executable expression within the expression.

```bash
$> symrun my_alias arg1 arg2 kwarg1=value1 kwarg2=value2
```

The following command creates a new alias named `my_alias` that points to `<username>/<repo_name>`:

```bash
$> symrun c my_alias <username>/<repo_name>
```

The following command lists all the aliases that have been created:

```bash
$> symrun l
```

The following command removes the alias named `my_alias`:

```bash
$> symrun r my_alias
```

#### Alias File

The Package Runner stores aliases in a JSON file named `aliases.json`. This file is located in the `.symai/packages/` directory in your home directory (`~/.symai/packages/`). You can view the contents of this file to see the existing aliases.

Here is an example how to use the `sympkg` and `symrun` via shell:
![Demo Usage of symask](https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/screen1.jpeg)

> [NOTE]: If the alias specified cannot be found in the alias file, the Package Runner will attempt to run the command as a package. If the package is not found or an error occurs during execution, an appropriate error message will be displayed.

That's it! You now have a basic understanding of how to use the Package Runner provided to run packages and aliases from the command line.

### 📦 Package Initializer

The Package Initializer is a command-line tool provided that allows developers to create new GitHub packages from the command line. It automates the process of setting up a new package directory structure and files. You can access the Package Initializer by using the `symdev` command in your terminal or PowerShell.

#### Usage

To use the Package Initializer, you can run the following command:

```bash
$> symdev c <username>/<repo_name>
```

The most commonly used Package Initializer command is:

- `c <username>/<repo_name>`: Create a new package

#### Examples

Here is an example to illustrate how to use the Package Initializer:

```bash
$> symdev c symdev/my_package
```

This command creates a new package named `my_package` under the GitHub username `symdev`.

The Package Initializer creates the following files and directories:

- `.gitignore`: Specifies files and directories that should be ignored by Git.
- `LICENSE`: Contains the license information for the package.
- `README.md`: Contains the description and documentation for the package.
- `requirements.txt`: Lists the packages and dependencies required by the package.
- `package.json`: Provides metadata for the package, including version, name, description, and expressions.
- `src/func.py`: Contains the main function and expression code for the package.

The Package Initializer creates the package in the `.symai/packages/` directory in your home directory (`~/.symai/packages/<username>/<repo_name>`).
Within the created package you will see the `package.json` config file defining the new package metadata and `symrun` entry point and offers the declared expression types to the `Import` class. 
Read more about the {doc}`import class <FEATURES/import>`