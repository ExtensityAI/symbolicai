import platform
import subprocess

from typing import Dict, List

from ..post_processors import CodeExtractPostProcessor
from ..symbol import Expression, Symbol

Context = """[DESCRIPTION]:
Adapt the user query to an OS patform command (commands must be executable in terminal, shell, bash or powershell)!
Create only command adaptations based on programs that are specified in the programs list or are native platform commands!

[PROGRAM_LIST]:

{programs}
For other command requests, reply sorry not supported.

[PLATFORM]:

The currently detected OS platform is:
{platform}
ONLY CREATE A COMMAND FOR THE CURRENT PLATFORM!

[USER_QUERY]:

{query}

[METADATA]:

Metadata is OPTIONAL and can help with the specificity of the command.
{metadata}

---------------------------

[EXAMPLES]:

If the current platform is Windows, and the user query is: "open file.txt", then the command should be:
```powershell
notepad file.txt
```
If the current platform is Linux, and the user query is: "open file.txt", then the command should be:
```shell
gedit file.txt
```
If the current platform is Mac, and the user query is: "open file.txt", then the command should be:
```bash
open file.txt
```
If the current platform is Windows, and the user query requires to open Spotify and play Taylor Swift, and Spotify is in the programs list, then the command could look like:
```powershell
Start-Process 'spotify:track:Anti-Hero%20by%20Taylor%20Swift'
```
If the current platform is Windows, and the user query requires to open Spotify and play a song, and Spotify is in the programs list, and metadata is added, then the command could look like:
```powershell
Start-Process 'spotify:track:0V3wPSX9ygBnCm8psDIegu?si=81646e6079d34526'
```

---------------------------

Write an executable command that starts a process according to the user query, platform and the programs list. The commnad should be one line and should be direcly executable in terminal, shell, bash or powershell.
"""


class OSCommand(Expression):
    def __init__(self, programs:        List[str],
                       metadata:   Dict[str, str] = {},
                       verbose:              bool = False,
                       os_platform:           str = 'auto',
                       **kwargs):
        super().__init__(**kwargs)
        self.verbose:            bool = verbose
        self.os_platform:         str = os_platform
        self.programs:      List[str] = programs
        self.meta: Dict[str, str]     = metadata

        if self.os_platform == 'auto':
            self.os_platform = platform.platform()
        if len(programs) == 0:
            raise Exception('No programs specified!')

    def execute_os_command(self, *args, **kwargs):
        command = args[0]
        print(f'Executing {self.os_platform} command: {command}')
        if 'linux' in self.os_platform.lower():
            return [subprocess.run(["bash", "-c", str(command)])]
        elif 'windows' in self.os_platform.lower():
            return [subprocess.run(["powershell", "-Command", str(command)])]
        elif 'mac' in self.os_platform.lower():
            return [subprocess.run(["bash", "-c", str(command)])]
        else:
            raise Exception('Unsupported platform!')

    def forward(self, sym: Symbol, **kwargs) -> Expression:
        sym = self._to_symbol(sym)
        kwargs['verbose'] = self.verbose

        prompt = Context.format(programs=self.programs,
                                platform=self.os_platform,
                                query=sym,
                                metadata=self.meta)
        command = sym.query(prompt, post_processors=[CodeExtractPostProcessor()], **kwargs)
        return self.sym_return_type(self.output(expr=self.execute_os_command, raw_input=True, processed_input=command.value, **kwargs))
