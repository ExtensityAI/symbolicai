import argparse
import json
import os
import sys
from pathlib import Path


class PackageInitializer():
    def __init__(self):
        self.package_dir = Path.home() / '.symai/packages/'

        if not os.path.exists(self.package_dir):
            os.makedirs(self.package_dir)

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
            setattr(args, 'package', args.command)
            self.c(args)
        elif len(args.command) == 1 and not hasattr(self, args.command):
            print('Unrecognized command')
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
        except:
            print('Invalid package name: {git_username}/{package_name}')
            parser.print_help()
            exit(1)

        package_path = os.path.join(self.package_dir, username, package_name)
        if os.path.exists(package_path):
            print('Package already exists')
            exit(1)

        print('Creating package...')
        os.makedirs(package_path)
        os.makedirs(os.path.join(package_path, 'src'))

        with open(os.path.join(package_path, '.gitignore'), 'w'): pass
        with open(os.path.join(package_path, 'LICENSE'), 'w') as f:
            f.write('MIT License')
        with open(os.path.join(package_path, 'README.md'), 'w') as f:
            f.write('# ' + package_name + '\n## <Project Description>')
        with open(os.path.join(package_path, 'requirements.txt'), 'w'): pass
        with open(os.path.join(package_path, 'package.json'), 'w') as f:
            json.dump({
                'version': '0.0.1',
                'name': username+'/'+package_name,
                'description': '<Project Description>',
                'expressions': [{'module': 'src/func', 'type': 'MyExpression'}],
                'run': {'module': 'src/func', 'type': 'MyExpression'},
                'dependencies': []
            }, f, indent=4)
        with open(os.path.join(package_path, 'src', 'func.py'), 'w') as f:
            f.write("""from symai import Expression, Function


FUNCTION_DESCRIPTION = '''
# TODO: Your function description here.
{template}
'''


class MyExpression(Expression):
    def __init__(self):
        super().__init__()
        self.fn = Function(FUNCTION_DESCRIPTION)

    def forward(self, data, template: str = '', *args, **kwargs):
        data = self._to_symbol(data)
        self.fn.format(template=template)
        return self.fn(data, *args, **kwargs)""")
            print('Package created successfully at: ' + package_path)


def run() -> None:
    PackageInitializer()


if __name__ == '__main__':
    run()
