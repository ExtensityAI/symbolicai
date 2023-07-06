import argparse
import json
import os
import sys
from pathlib import Path
from symai import Import
from colorama import Fore, Style


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
                usage='''symrun <alias> [<args>] | <command> <alias> [<package>]'''
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

    def console(self, alias, package, output: str):
        print(Fore.GREEN + Style.BRIGHT + "Execution of {}::{} resulted in the following output:".format(alias, package))
        print(Fore.MAGENTA + Style.DIM + str(output))
        print(Style.RESET_ALL)

    def run_alias(self):
        parser = argparse.ArgumentParser(
            description='This will create a new alias entry in the alias json file. Exsisting aliases will be overwritten.',
            usage='symrun <alias> [<args> | <kwargs>]*'
        )
        parser.add_argument('alias', help='Name of alias to run')
        parser.add_argument('params', nargs=argparse.REMAINDER)
        args = parser.parse_args(sys.argv[1:])

        aliases = self.load_aliases()
        package = aliases.get(args.alias)

        if package is None:
            print(Fore.RED + Style.BRIGHT + "Alias run of `{}` not found. Please check your command {}".format(args.alias, args))
            print(Style.RESET_ALL)
            parser.print_help()
            return
        expr = Import(package)

        arg_values = [arg for arg in args.params if '=' not in arg]
        kwargs = {arg.split('=')[0]: arg.split('=')[1] for arg in args.params if '=' in arg}

        result = expr(*arg_values, **kwargs)
        self.console(args.alias, package, result)
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


def run() -> None:
    PackageRunner()


if __name__ == '__main__':
    run()