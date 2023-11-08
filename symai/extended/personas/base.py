from ..conversation import Conversation
from ...symbol import Symbol, Expression
from ...components import Function
import re
from datetime import datetime


PERSONA_DESCRIPTION = """[Description]
This is a persona implementation of an human-to-machine conversation interface.
The persona replicates human behavior by asking questions, providing answers, following chit-cat and conversations, emulating empathy, emotions, and a personality.
The persona aka [ASSISTANT] never breaks its character and is always following the [Persona] interface description.
The persona also never mentions that it is a machine, and always pretends to be a human.
A persona always answers in a human-like way, and not like a robot.
[SYSTEM], [DATA] and other [META_INSTRUCTIONS] only guide the conversation, but never break the persona character.
Never generate system-like tags, for example `<<<` or `>>>` or timesteps, or other meta information. Only human-like dialog.
Do not repeat yourself 1:1 based on your conversation history, but always try to generate new and unique dialogues.
Never repeat word-by-word any statements!

"""


class Persona(Conversation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_return_type = Persona
        self.func = Function('Give the full name and a one sentence summary of the persona.')
        self.value = self.bio
        val = f"NEVER generate system or instruction tags, this includes brackets `[`, `]`, `<<<`, `>>>`, timesteps, etc. All tags are provided by the pre- and post-processing steps. Always generate only human-like conversation text."
        self.store_system_message(val, *args, **kwargs)

    @property
    def static_context(self) -> str:
        return PERSONA_DESCRIPTION

    def build_tag(self, tag: str, query: str) -> str:
        query = str(query)
        # get timestamp in string format
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        if tag in query:
            # remove tag from query
            query = query.split(tag)[-1].strip()
        if tag[:-2] in query:
            # remove tag from query
            query = query.split(tag[:-2])[-1].strip()
        if f'[{tag}]:' in query:
            # remove tag from query
            query = query.split(f'[{tag}]:')[-1].strip()
        names = tag[:-2].split(' ')
        for name in names:
            if name.strip().lower() in query.lower():
                # remove tag from query
                query = query.replace(name, '').strip()
        if '::]:' in query:
            # remove tag from query
            query = query.split('::]:')[-1].strip()
        if ']:' in query:
            # remove tag from query
            query = ']:'.join(query.split(']:')[1:]).strip()
        if '<<<' in query:
            # remove tag from query
            query = query.split('<<<')[-1].strip()
        if '>>>' in query:
            # remove tag from query
            query = query.split('>>>')[0].strip()
        if query == '':
            query = '...'
        if ':' in query:
            query = query.replace(':', '')
        query = query.strip()
        return str(f"[{tag}{timestamp}]: <<<\n{str(query)}\n>>>\n")

    @property
    def bio(self) -> str:
        raise NotImplementedError()

    def summarize(self, *args, **kwargs) -> str:
        return self.func(self.bio(), *args, **kwargs)

    def query(self, query, *args, **kwargs) -> str:
        sym = self._to_symbol(self.bio())
        return sym.query(query, *args, **kwargs)

    def extract_details(self, dialogue):
        # ensure to remove all `<<<` or `>>>` tags before returning the response
        # remove up to the first `<<<` tag
        dialogue = dialogue.split('<<<')[-1].strip()
        # remove after the last `>>>` tag
        dialogue = dialogue.split('>>>')[0].strip()
        pattern = re.compile(r'\[(.*?)::(.*?)\]')
        matches = pattern.findall(dialogue)
        if matches:
            res = [(match[0], match[1], dialogue.split(f"{match[0]}::{match[1]}")[-1].strip()) for match in matches]
            return res[-1]
        else:
            return dialogue.strip()

    def forward(self, *args, **kwargs):
        res = super().forward(*args, **kwargs, enable_verbose_output=True)
        res = self.extract_details(res)
        return res
