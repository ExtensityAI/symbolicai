import argparse
import os
import pprint
import sys
from pathlib import Path

from loguru import logger

from ... import config_manager
from ...imports import Import


class PackageHandler():
    def __init__(self):
        self.package_dir = Path(config_manager.config_dir) / 'packages'

        if not os.path.exists(self.package_dir):
            os.makedirs(self.package_dir)

        os.chdir(self.package_dir)

        parser = argparse.ArgumentParser(
            description='''SymbolicAI package manager.
            Manage extensions from the command line.
            You can (i) install, (r) remove, (l) list installed, (u) update a module or (U) update all modules.''',
            usage='''sympkg <command> [<args>]

            The most commonly used sympkg commands are:
            i   Install a new package (--local-path PATH, --submodules)
            r   Remove an installed package
            l   List all installed packages
            u   Update an installed package (--submodules)
            U   Update all installed packages (--submodules)
            '''
        )

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if len(args.command) > 1 and not hasattr(self, args.command):
            setattr(args, 'package', args.command)
            self.i(args)
        elif len(args.command) == 1 and not hasattr(self, args.command):
            logger.error('Unrecognized command')
            parser.print_help()
            exit(1)
        else:
            getattr(self, args.command)()

    def i(self, args = None):
        parser = argparse.ArgumentParser(
            description='Install a new package',
            usage='sympkg i [package] [--local-path PATH] [--submodules]'
        )
        parser.add_argument('package', help='Name of package to install')
        parser.add_argument('--local-path', '-l', help='Local path to package directory')
        parser.add_argument('--submodules', '-s', action='store_true', help='Initialize submodules for GitHub repos')

        if args is None:
            args = parser.parse_args(sys.argv[2:])

        Import.install(args.package, args.local_path, args.submodules)

    def r(self):
        parser = argparse.ArgumentParser(
            description='Remove an installed package',
            usage='sympkg r [package]'
        )
        parser.add_argument('package', help='Name of package to remove')
        args = parser.parse_args(sys.argv[2:])
        Import.remove(args.package)

    def l(self):
        pprint.pprint(Import.list_installed())

    def u(self):
        parser = argparse.ArgumentParser(
            description='Update an installed package',
            usage='sympkg u [package] [--submodules]'
        )
        parser.add_argument('package', help='Name of package to update')
        parser.add_argument('--submodules', '-s', action='store_true', help='Update submodules as well')
        args = parser.parse_args(sys.argv[2:])
        Import.update(args.package, args.submodules)

    def U(self):
        parser = argparse.ArgumentParser(
            description='Update all installed packages',
            usage='sympkg U [--submodules]'
        )
        parser.add_argument('--submodules', '-s', action='store_true', help='Update submodules as well')
        args = parser.parse_args(sys.argv[2:])

        packages = Import.list_installed()
        for package in packages:
            try:
                logger.info(f'[UPDATE]: Updating {package}...')
                Import.update(package, args.submodules)
            except Exception as e:
                logger.error(f'[SKIP]: Error updating {package}: {e}')


def run() -> None:
    PackageHandler()


if __name__ == '__main__':
    run()
