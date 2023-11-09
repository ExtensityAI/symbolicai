from ...components import Function
from ...symbol import Expression, Symbol
from . import Persona # keep this import for reflection to work


PERSONA_BUILDER_DESCRIPTION = """[Task]
Create a detailed persona description about a character called {name}, which is as {description}.
Be very detailed and specific about the persona, and include his visual appearance, character description, personality, background, quirks, and other relevant information. Add also past job and education history.
Add also a description how the persona usually interacts with other people, and how he speaks and behaves in general.
Include also friends and family and how the persona interacts with them. Provide at least two names of his family or close friends and a one sentence description of them. {relation_description}

[Exemplary Output Structure] // add and vary the information or bullets if needed
Persona Description >>>
Name: ...
Age: ...
Height: ...
Build: ...
Hair Color: ...
Eye Color: ...
Fashion Sense: ...
... # other relevant information

Character Description:
...

Personality:
...

Background:
...

Education:
...

Quirks:
...

Interactions with Other People:
...

Friends and Family:
...

Past Job and Education History:
...

Additional Information:
...
<<<
"""


PERSONA_TEMPLATE = """
def run():
    from symai.extended.personas import Persona

    CURRENT_PERSONA_DESCRIPTION = '''{persona_description}'''

    class PersonaClass(Persona):
        @property
        def static_context(self) -> str:
            return super().static_context + CURRENT_PERSONA_DESCRIPTION

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.bot_tag         = f'{name}::'
            self.user_tag        = 'Other Person::'
            self.sym_return_type = PersonaClass

        def bio(self) -> str:
            return CURRENT_PERSONA_DESCRIPTION

    return PersonaClass
"""


class PersonaBuilder(Expression):
    def __init__(self):
        super().__init__()
        self.func = Function(PERSONA_BUILDER_DESCRIPTION)

    def forward(self, name: str, description: str, relation: str = '', resume: Symbol = None, **kwargs) -> str:
        if relation:
            relation = f'The persona is related to {relation}.'
        self.func.format(name=name, description=description, relation_description=relation, **kwargs)
        if resume:
            sym = '[PERSONA GENERATION GUIDANCE based on RESUME]\n' @ self._to_symbol(resume)
            description = self.func(payload=sym)
        else:
            description = self.func()
        # extract full name
        full_name = None
        for line in description.split('\n'):
            if 'Name:' in line:
                full_name = line.split('Name:')[-1].strip()
                break
        class_node = PERSONA_TEMPLATE.format(persona_description=description, name=full_name)
        # Create the class dynamically and return an instance.
        globals_ = globals().copy()
        locals_ = {}
        exec(class_node, globals_, locals_)
        PersonaClass = locals_['run']()
        return PersonaClass
