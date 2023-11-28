import json
import os
import sys
import argparse

from pathlib import Path
from typing import Optional

from ...imports import Import
from ...misc.console import ConsoleStyle
from ...misc.loader import Loader


class PackageRunner():
    def __init__(self):
        self.package_dir = Path.home() / '.symai/packages/'
        self.aliases_file = self.package_dir / 'aliases.json'

        if not os.path.exists(self.package_dir):
            os.makedirs(self.package_dir)

        os.chdir(self.package_dir)

        try:
            parser = argparse.ArgumentParser(
                description='''SymbolicAI package runner.
                Run a package in command line.''',
                usage='''symrun <alias> [<args>] | <command> <alias> [<package>]
                The most commonly used symrun commands are:
                <alias> [<args>]    Run an alias
                c <alias> <package> Create a new alias
                l                   List all aliases
                r <alias>           Remove an alias
'''
            )

            parser.add_argument('command', help='Subcommand to run')
            args = parser.parse_args(sys.argv[1:2])
            getattr(self, args.command)()
        except:
            if len(sys.argv) > 1:
                self.run_alias()
            else:
                parser.print_help()
                exit(1)

    def load_aliases(self):
        if not os.path.exists(self.aliases_file):
            return {}

        with open(self.aliases_file, 'r') as f:
            return json.load(f)

    def save_aliases(self, aliases):
        with open(self.aliases_file, 'w') as f:
            json.dump(aliases, f)

    def console(self, header: str, output: Optional[object] = None):
        with ConsoleStyle('success'):
            print(header)
        if output is not None:
            with ConsoleStyle('info'):
                print(str(output))

    def run_alias(self):
        parser = argparse.ArgumentParser(
            description='This command runs the alias from the aliases.json file. If the alias is not found, it will run the command as a package.',
            usage='symrun <alias> [<args> | <kwargs>]*'
        )
        parser.add_argument('alias', help='Name of alias to run')
        parser.add_argument('params', nargs=argparse.REMAINDER)
        args = parser.parse_args(sys.argv[1:])

        aliases = self.load_aliases()
        # try running the alias or as package
        package = aliases.get(args.alias) or args.alias

        if package is None:
            with ConsoleStyle('error'):
                print("Alias run of `{}` not found. Please check your command {}".format(args.alias, args))
            parser.print_help()
            return

        try:
            expr = Import(package)
        except Exception as e:
            with ConsoleStyle('error'):
                print("Error: {} in package `{}`.\nPlease check your command {} or if package is available.".format(str(e), package, args))
            parser.print_help()
            return

        arg_values = [arg for arg in args.params if '=' not in arg]
        kwargs = {arg.split('=')[0]: arg.split('=')[1] for arg in args.params if '=' in arg}

        with Loader(desc="Inference ...", end=""):
            result = expr(*arg_values, **kwargs)
        self.console("Execution of {} => {} resulted in the following output:".format(args.alias, package), result)
        return result

    def c(self):
        parser = argparse.ArgumentParser(
            description='This will create a new alias entry in the alias json file. Exsisting aliases will be overwritten.',
            usage='symrun c <alias> <package>'
        )
        parser.add_argument('alias', help='Name of user based on GitHub username and package to install')
        parser.add_argument('package', help='Name of the package: <user>/<package> where user is based on GitHub username and package to install')
        args = parser.parse_args(sys.argv[2:])

        aliases = self.load_aliases()
        aliases[args.alias] = args.package
        self.save_aliases(aliases)
        self.console("Alias {} => {} created successfully.".format(args.alias, args.package))

    def l(self):
        aliases = self.load_aliases()
        # format the aliases output as a table of key value pairs
        self.console("Aliases:\n------------------")
        for alias, package in aliases.items():
            self.console(f'{alias} => {package}')
        self.console("------------------")

    def r(self):
        parser = argparse.ArgumentParser(
            description='This will remove the alias name from the alias json file.',
            usage='symrun r <alias>'
        )
        parser.add_argument('alias', help='Name of alias to remove')
        args = parser.parse_args(sys.argv[2:])

        aliases = self.load_aliases()
        if args.alias in aliases:
            del aliases[args.alias]
        self.save_aliases(aliases)
        self.console("Alias {} => {} removed.".format(args.alias, args.package))


def run() -> None:
    PackageRunner()


if __name__ == '__main__':
    run()