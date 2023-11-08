from ...components import Function
from ...symbol import Expression
from .base import Persona


PERSONA_BUILDER_DESCRIPTION = """[Task]
Create a detailed persona description about a character called {name}, which is as {description}.
Be very detailed and specific about the persona, and include his visual appearance, character description, personality, background, quirks, and other relevant information. Add also past job and education history.
Add also a description how the persona usually interacts with other people, and how he speaks and behaves in general.
Include also friends and family and how the persona interacts with them. Provide at least two names of his family or close friends and a one sentence description of them. {relation_description}
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
            self.sym_return_type = Persona

        def bio(self) -> str:
            return CURRENT_PERSONA_DESCRIPTION

    return PersonaClass
"""


class PersonaBuilder(Expression):
    def __init__(self):
        super().__init__()
        self.func = Function(PERSONA_BUILDER_DESCRIPTION)

    def forward(self, name: str, description: str, relation: str = '', **kwargs) -> str:
        if relation:
            relation = f'The persona is related to {relation}.'
        self.func.format(name=name, description=description, relation_description=relation, **kwargs)
        description = self.func()
        class_node = PERSONA_TEMPLATE.format(persona_description=description)
        # Create the class dynamically and return an instance.
        globals_ = globals().copy()
        locals_ = {}
        exec(class_node, globals_, locals_)
        PersonaClass = locals_['run']()
        return PersonaClass
