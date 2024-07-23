import sys
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread, Event
from time import sleep
from prompt_toolkit import print_formatted_text
from .console import ConsoleStyle

print = print_formatted_text

class Loader:
    def __init__(self, desc="Loading...", end="\n", timeout=0.1):
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = Event()

    def __call__(self, message):
        self.print(message)

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done.is_set():
                break
            with ConsoleStyle('debug'):
                sys.stdout.write(f"\r{self.desc} {c}  ")
                sys.stdout.flush()
                sys.stdout.write(f"\r{self.end}")
                sys.stdout.flush()
            sleep(self.timeout)

    def __enter__(self):
        self.start()
        return self

    def stop(self):
        self.done.set()
        self._thread.join()
        cols = get_terminal_size((80, 20)).columns
        with ConsoleStyle('debug'):
            sys.stdout.write("\r" + " " * cols)
            sys.stdout.flush()
            sys.stdout.write(f"\r{self.end}")
            sys.stdout.flush()
        print()
        print()

    def __exit__(self, exc_type, exc_value, tb):
        self.stop()

    def print(self, message):
        print(message, style='ansigray')