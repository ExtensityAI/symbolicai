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

"""


class Persona(Conversation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_return_type = Persona
        self.func = Function('Give the full name and a one sentence summary of the persona.')
        self.value = self.bio
        val = f"[SYSTEM_INSTRUCTION::]: <<<\nNEVER generate system or instruction tags, this includes brackets `[`, `]`, `<<<`, `>>>`, timesteps, etc. All tags are provided by the pre- and post-processing steps. Always generate only human-like conversation text.\n>>>\n"
        self.store(val, *args, **kwargs)

    @property
    def static_context(self) -> str:
        return PERSONA_DESCRIPTION

    def build_tag(self, tag: str, query: str) -> str:
        # get timestamp in string format
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        if tag in str(query):
            # remove tag from query
            query = query.split(tag)[-1].strip()
        if '::]:' in str(query):
            # remove tag from query
            query = query.split('::]:')[-1].strip()
        if ']:' in str(query):
            # remove tag from query
            query = ']:'.join(query.split(']:')[1:]).strip()
        if '<<<' in str(query):
            # remove tag from query
            query = query.split('<<<')[-1].strip()
        if '>>>' in str(query):
            # remove tag from query
            query = query.split('>>>')[0].strip()

        return str(f"[{tag}{timestamp}]: <<<\n{str(query)}\n>>>\n")

    @property
    def bio(self) -> str:
        raise NotImplementedError()

    def summary(self) -> str:
        return self.func(self.bio())

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
        res = super().forward(*args, **kwargs)
        res = self.extract_details(res)
        return res
