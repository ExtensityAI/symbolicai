from pathlib import Path

from symai.prompts import PromptRegistry

PROMPTS_DIR = Path(__file__).parent

prompt_registry = PromptRegistry()
prompt_registry.load_from_folder(PROMPTS_DIR)


def render_chat_system_prompt(argument) -> str:
    prop = argument.prop
    response_format_type = None

    if prop.response_format:
        try:
            response_format_type = prop.response_format["type"]
        except KeyError as e:
            msg = 'Response format type is required! Expected format `{"type": "json_object"}` or other supported types.'
            raise AssertionError(msg) from e
        if response_format_type is None:
            msg = 'Response format type is required! Expected format `{"type": "json_object"}` or other supported types.'
            raise AssertionError(msg)

    static_context, dynamic_context = prop.instance.global_context
    return prompt_registry.render(
        "system",
        suppress_verbose_output=prop.suppress_verbose_output,
        response_format_type=response_format_type,
        static_context=static_context,
        dynamic_context=dynamic_context,
        payload=prop.payload,
        examples=prop.examples,
        prompt=prop.prompt,
        template_suffix=prop.template_suffix,
    )
