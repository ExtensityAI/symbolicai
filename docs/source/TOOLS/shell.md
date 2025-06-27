# Shell Command Tool

The Shell Command Tool is a basic shell command support tool that translates natural language commands into shell commands. To start the Shell Command Tool, simply run:

```bash
symsh "<your-query>"
```

For more information about the tool and available arguments, use the `--help` flag:

```bash
symsh --help
```

Here is an example of how to use the Shell Command Tool:

```bash
$> symsh "PowerShell edit registry entry"

# :Output:
# Set-ItemProperty -Path <path> -Name <name> -Value <value>

$> symsh "Set-ItemProperty -Path <path> -Name <name> -Value <value>" --add "path='/Users/myuser' name=Demo value=SymbolicAI"

# :Output:
# Set-ItemProperty -Path '/Users/myuser' -Name Demo -Value SymbolicAI

$> symsh "Set-ItemProperty -Path '/Users/myuser' -Name Demo -Value SymbolicAI" --del "string quotes"

# :Output:
# Set-ItemProperty -Path /Users/myuser -Name Demo -Value SymbolicAI

$> symsh "Set-ItemProperty -Path '/Users/myuser' -Name Demo -Value SymbolicAI" --convert "linux"

# :Output:
# export Demo="SymbolicAI"
```


## 🖥️ Interactive Shell

`symsh` is also a regular shell program that interacts with users in the terminal emulation window. It interprets Linux, MacOS, and Windows PowerShell shell commands, and supports ANSI escape sequences.

> ❗️**NOTE**❗️Because the colors for the default style is highly dependent on whether the theme is light or dark, they may not be displayed correctly in some terminals. You can change the default style to better fit your needs by modifying the `symsh.config.json` file in the `.symai` directory in your home directory (`~/.symai/symsh.config.json`).

To enter an interactive shell, simply run without any additional parameters:

```bash
$> symsh
```

The interactive shell uses the `python -m symai.shell` feature and runs on top of your existing terminal.

Within the interactive shell you can use your regular shell commands and additionally use the `symsh` neuro-symbolic commands. The interactive shell supports the following commands:

### Auto-completion
`symsh` provides path auto-completion and history auto-completion enhanced by the neuro-symbolic engine. Start typing the path or command, and `symsh` will provide you with relevant suggestions based on your input and command history.
To trigger a suggestion, press `Tab` and then `Enter`.

![Demo usage of symsh](https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/symsh.png)

### Query Neuro-Symbolic Model
`symsh` can interact with a language model. By beginning a command with a special character (`"`, `'`, or `` ` ``), `symsh` will treat the command as a query for a language model.

For instance, to make a query, you can type:

```bash
$> "What is the capital of France?"

# :Output:
# Paris
```
You can also type `Ctrl+Space` to treat any command as a query for a language model.

### Pipe with Files

The shell command in `symsh` also has the capability to interact with files using the pipe (`|`) operator. It operates like a Unix-like pipe but with a few enhancements due to the neuro-symbolic nature of `symsh`.

Here is the basic usage of the pipe with files:

```bash
$> "explain this file" | file_path.txt
```
This command would instruct the AI to explain the file `file_path.txt` and consider its contents for the conversation.

### Pipe with Commands
`symsh` can also interact with other shell commands using the pipe (`|`) operator. This allows you to execute a shell command and use its output as input to the language model along with your query.
**Basic Usage:**
```bash
$> "Your query" | command [arguments]
```
**Example:**
Suppose you want to understand the usage of a complex command like `ffmpeg`, which has extensive help documentation. Instead of manually reading through the lengthy help output, you can ask `symsh` to summarize it for you:
```bash
$> "Summarize how to convert a video using ffmpeg" | ffmpeg -h
```
This command runs `ffmpeg -h`, captures its output, and then asks the language model to provide a concise summary.
Similarly, if you're unsure about the options available for `grep`, which can be tedious to read through:
```bash
$> "Explain the options available for grep" | grep --help
```
This will execute `grep --help` and pass the output to the language model, which will then explain the various options in an understandable manner.
These additions explain how to use the new feature that allows piping queries with commands, provide examples of its usage, and outline the current limitations.

### Slicing Operation on Files
The real power of `symsh` shines through when dealing with large files. `symsh` extends the typical file interaction by allowing users to select specific sections or slices of a file.

To use this feature, you would need to append the desired slices to the filename within square brackets `[]`. The slices should be comma-separated, and you can apply Python's indexing rules. You can specify a single line, a range of lines, or step indexing.

Here are a few examples:

Single line:

```bash
$> "analyze this line" | file_path.txt[10]
```

Range of lines:

```bash
# analyze lines 10 to 20
$> "analyze this line" | file_path.txt[10:20]
```

Step indexing:

```bash
# analyze lines 10 to 30 with a step size of 3
$> "analyze this line" | file_path.txt[10:30:3]
```

Multi-line indexing:

```bash
# analyze lines 10 to 30 with a step size of 3, and lines 40 to 50
$> "analyze this line" | file_path.txt[10:30:3,20,40:50]
```

The above commands would read and include the specified lines from file `file_path.txt` into the ongoing conversation.

This feature enables you to maintain highly efficient and context-thoughtful conversations with `symsh`, especially useful when dealing with large files where only a subset of content in specific locations within the file is relevant at any given moment.

### Stateful Conversation

The stateful_conversation feature is used for maintaining a continuing conversation with the language model. To use this feature, you have to start your commands with specific symbols in the shell:

1. Creating a new stateful conversation:

   Use any of these three symbols at the start of your command: `!"`, `!'`, or `` !` ``. This will initialize a new stateful conversation. If there was a previously saved conversation, these commands will overwrite it.

2. Continuing a stateful conversation:

   Use one of these three symbols at the start of your command: `."`, `.'`, or `` .` ``. The command can then be used to continue the most recent stateful conversation. If no previous conversation exists, a new one is created.

Example:

   1. Starting a new conversation: `!"what is your name"`
   2. Continuing the conversation: `."how old are you"`

These commands can be used in any shell operation. Keep in mind, stateful conversations are saved and can be resumed later. The shell will save the conversation automatically if you type `exit` or `quit` to exit the interactive shell.

Stateful conversation offers the capability to process files as well. If your command contains a pipe (`|`), the shell will treat the text after the pipe as the name of a file to add it to the conversation.

Example:
```bash
$> !"explain this file" | my_file.txt
```

This command will instruct the AI to explain the file `my_file.txt` and consider its contents in the conversation. Afterwards you can continue the conversation with:

```bash
$> ."what did you mean with ...?"
```

### Limitations and Notes
Currently, `symsh` supports piping your query to either a command or file(s), but **not both simultaneously**. The following cases are supported:
- **Query with Command**: `query | command [arguments]`
  *Example*:
  ```bash
  $> "How can I use the options for tar?" | tar --help
  ```
- **Query with File(s)**: `query | file [file ...]`
  *Example*:
  ```bash
  $> "Summarize the contents of these files" | file1.txt file2.txt
  ```
The following cases are **not supported** and will result in an error:
1. **Query with Command and File(s)**: `query | command | file`
   *Not Supported*:
   ```bash
   $> "Process this data" | awk '{print $1}' | data.txt
   ```
2. **Query with Multiple Commands**: `query | command1 | command2`
   *Not Supported*:
   ```bash
   $> "Explain the output" | ls -l | sort
   ```
3. **Query with File(s) and Command**: `query | file | command`
   *Not Supported*:
   ```bash
   $> "Analyze this data" | data.txt | sort
   ```
> **Note**: If you attempt to use unsupported combinations, `symsh` will raise an error and prompt you to adjust your command accordingly. Ensure that your command outputs text to standard output (stdout). Binary outputs or commands that do not produce textual output may not work as expected.
