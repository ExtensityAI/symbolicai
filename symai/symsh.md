"LLM Command Usage and Detail:

llm (language model) processing is initiated with specific start flags `'`, `"` and `` ` `` and optionally preceded by `.`, `!`, or `?`.

1. Normal Mode: `'`, `"` and `` ` ``
   Usage: 'Your Query' or "Your Query" or `Your Query`
   These commands send 'Your Query' to sym's language model and return the result.

2. Stateful Mode: `.'`, `."` and `.`
   Usage: .'Your Query' or ."Your Query" or .`Your Query`
   These commands create a stateful conversation with sym’s language models. Once you have initiated a stateful conversation
   with `.`, continuing with queries using `.` will result in the conversation maintaining a context from all previous queries
   and responses.

3. Overwrite Mode: `!'`, `!"`, `!`
   Usage: !'Your Query' or !"Your Query" or !`Your Query`
   These commands function as their normal or stateful counterparts, but also overwrite the existing conversation_state.pickle.

4. Search Mode: `?'`, `?"`, `?`
   Usage: ?'Your Query' or ?"Your Query" or ?`Your Query`
   This command sends 'Your Query' to sym's internal search engine and the results are returned in the shell.

Note: The use of `'`, `"` or `` ` `` to wrap 'Your Query' does not impact the functionality of the command."