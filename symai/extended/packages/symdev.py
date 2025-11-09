import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

from ... import config_manager


class PackageInitializer:
    def __init__(self):
        self.package_dir = Path(config_manager.config_dir) / 'packages'

        if not self.package_dir.exists():
            self.package_dir.mkdir(parents=True)

        os.chdir(self.package_dir)

        parser = argparse.ArgumentParser(
            description='''SymbolicAI package initializer.
            Initialize a new GitHub package from the command line.''',
            usage='''symdev <command> <username>/<package_name>
            Available commands:
            c   Create a new package [default if no command is given]
'''
        )

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if len(args.command) > 1 and not hasattr(self, args.command):
            args.package = args.command
            self.c(args)
        elif len(args.command) == 1 and not hasattr(self, args.command):
            logger.error('Unrecognized command')
            parser.print_help()
            exit(1)
        else:
            getattr(self, args.command)()

    def c(self, args = None):
        parser = argparse.ArgumentParser(
            description='Create a new package',
            usage='symdev c <username>/<package>'
        )
        parser.add_argument('package', help='Name of user based on GitHub username and package to install')
        if args is None:
            args = parser.parse_args(sys.argv[2:3])
        vals = args.package.split('/')
        try:
            username = vals[0]
            package_name = vals[1]
        except IndexError:
            logger.error('Invalid package name: {git_username}/{package_name}')
            parser.print_help()
            exit(1)

        package_path = self.package_dir / username / package_name
        if package_path.exists():
            logger.info('Package already exists')
            exit(1)

        logger.info('Creating package...')
        package_path.mkdir(parents=True)
        src_path = package_path / 'src'
        src_path.mkdir(parents=True)

        with (package_path / '.gitignore').open('w'):
            pass
        with (package_path / 'LICENSE').open('w') as f:
            f.write('MIT License')
        with (package_path / 'README.md').open('w') as f:
            f.write('# ' + package_name + '\n## <Project Description>')
        with (package_path / 'requirements.txt').open('w'):
            pass
        with (package_path / 'package.json').open('w') as f:
            json.dump({
                'version': '0.0.1',
                'name': username+'/'+package_name,
                'description': '<Project Description>',
                'expressions': [{'module': 'src/func', 'type': 'MyExpression'}],
                'run': {'module': 'src/func', 'type': 'MyExpression'},
                'dependencies': []
            }, f, indent=4)
        with (src_path / 'func.py').open('w') as f:
            f.write("""from symai import Expression, Function


FUNCTION_DESCRIPTION = '''
# TODO: Your function description here.
{template}
'''


class MyExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fn = Function(FUNCTION_DESCRIPTION)

    def forward(self, data, template: str = '', *args, **kwargs):
        data = self._to_symbol(data)
        self.fn.format(template=template)
        return self.fn(data, *args, **kwargs)""")
            logger.success('Package created successfully at: ' + str(package_path))


def run() -> None:
    PackageInitializer()


if __name__ == '__main__':
    run()
