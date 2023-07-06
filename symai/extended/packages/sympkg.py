import argparse
import os
import sys
from pathlib import Path

from ... import Import


class PackageHandler():
    def __init__(self):
        self.package_dir = Path.home() / '.symai/packages/'

        if not os.path.exists(self.package_dir):
            os.makedirs(self.package_dir)

        os.chdir(self.package_dir)

        parser = argparse.ArgumentParser(
            description='''SymbolicAI package manager.
            Manage extensions from the command line.
            You can (i) install, (r) remove, (l) list installed, or (u) update a module.''',
            usage='''sympkg <command> [<args>]

            The most commonly used sympkg commands are:
            i   Install a new package
            r   Remove an installed package
            l   List all installed packages
            u   Update an installed package
            '''
        )

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if len(args.command) > 1 and not hasattr(self, args.command):
            setattr(args, 'package', args.command)
            self.i(args)
        elif len(args.command) == 1 and not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        else:
            getattr(self, args.command)()

    def i(self, args = None):
        parser = argparse.ArgumentParser(
            description='Install a new package',
            usage='sympkg i [package]'
        )
        parser.add_argument('package', help='Name of package to install')
        if args is None:
            args = parser.parse_args(sys.argv[2:])
        Import.install(args.package)

    def r(self):
        parser = argparse.ArgumentParser(
            description='Remove an installed package',
            usage='sympkg r [package]'
        )
        parser.add_argument('package', help='Name of package to remove')
        args = parser.parse_args(sys.argv[2:])
        Import.remove(args.package)

    def l(self):
        import pprint
        pprint.pprint(Import.list_installed())

    def u(self):
        parser = argparse.ArgumentParser(
            description='Update an installed package',
            usage='sympkg u [package]'
        )
        parser.add_argument('package', help='Name of package to update')
        args = parser.parse_args(sys.argv[2:])
        Import.update(args.package)


def run() -> None:
    PackageHandler()


if __name__ == '__main__':
    run()
