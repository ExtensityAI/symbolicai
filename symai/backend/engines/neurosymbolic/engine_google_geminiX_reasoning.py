import base64
import logging
import mimetypes
import re
from copy import deepcopy
from pathlib import Path

import requests
from google import genai
from google.genai import types

from ....components import SelfPrompt
from ....utils import UserMessage, encode_media_frames
from ...base import Engine
from ...mixin.google import GoogleMixin
from ...settings import SYMAI_CONFIG

logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google_genai").propagate = False
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class TokenizerWrapper:
    def __init__(self, compute_tokens_func):
        self.compute_tokens_func = compute_tokens_func

    def encode(self, text: str) -> int:
        return self.compute_tokens_func([{"role": "user", "content": text}])

class GeminiXReasoningEngine(Engine, GoogleMixin):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if api_key is not None and model is not None:
            self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] = api_key
            self.config['NEUROSYMBOLIC_ENGINE_MODEL'] = model
        if self.id() != 'neurosymbolic':
            return # do not initialize if not neurosymbolic; avoids conflict with llama.cpp check in EngineRepository.register_from_package

        self.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.model = self.config['NEUROSYMBOLIC_ENGINE_MODEL']
        self.name = self.__class__.__name__
        self.tokenizer = TokenizerWrapper(self.compute_required_tokens)
        self.max_context_tokens = self.api_max_context_tokens()
        self.max_response_tokens = self.api_max_response_tokens()
        self.client = genai.Client(api_key=self.api_key)

    def id(self) -> str:
        model = self.config.get('NEUROSYMBOLIC_ENGINE_MODEL')
        if model and model.startswith('gemini'):
            return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
            genai.configure(api_key=self.api_key)
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']
            self.client = genai.GenerativeModel(model_name=self.model)

    def compute_required_tokens(self, messages) -> int:
        api_contents: list[types.Content] = []

        for msg in messages:
            msg_parts = msg if isinstance(msg, list) else [msg]
            for part in msg_parts:
                if isinstance(part, str):
                    role = 'user'
                    content_str = part
                elif isinstance(part, dict):
                    role = part.get('role')
                    content_str = str(part.get('content', ''))
                current_message_api_parts: list[types.Part] = []
                image_api_parts = self._handle_image_content(content_str)
                current_message_api_parts.extend(image_api_parts)

                text_only_content = self._remove_media_patterns(content_str)
                if text_only_content:
                    current_message_api_parts.append(types.Part(text=text_only_content))

                if current_message_api_parts:
                    api_contents.append(types.Content(role=role, parts=current_message_api_parts))

        if not api_contents:
            return 0

        try:
            count_response = self.client.models.count_tokens(model=self.model, contents=api_contents)
            return count_response.total_tokens
        except Exception as e:
            UserMessage(f"Gemini count_tokens failed: {e}")
            UserMessage(f"Error counting tokens for Gemini: {e!s}", raise_with=RuntimeError)

    def compute_remaining_tokens(self, _prompts: list) -> int:
        UserMessage("Token counting not implemented for Gemini", raise_with=NotImplementedError)

    def _handle_document_content(self, content: str):
        """Handle document content by uploading to Gemini"""
        try:
            pattern = r'<<document:(.*?):>>'
            matches = re.findall(pattern, content)
            if not matches:
                return None

            doc_path = matches[0].strip()
            if doc_path.startswith('http'):
                UserMessage("URL documents not yet supported for Gemini")
                return None
            return genai.upload_file(doc_path)
        except Exception as e:
            UserMessage(f"Failed to process document: {e}")
            return None

    def _handle_image_content(self, content: str) -> list[types.Part]:
        """Handle image content by processing and preparing google.generativeai.types.Part objects."""
        image_parts: list[types.Part] = []
        for img_src in self._extract_image_sources(content):
            try:
                image_parts.extend(self._create_parts_from_image_source(img_src))
            except Exception as e:
                UserMessage(f"Failed to process image source '{img_src}'. Error: {e!s}", raise_with=ValueError)
        return image_parts

    def _extract_image_sources(self, content: str) -> list[str]:
        pattern = r'<<vision:(.*?):>>'
        return [match.strip() for match in re.findall(pattern, content)]

    def _create_parts_from_image_source(self, img_src: str) -> list[types.Part]:
        if img_src.startswith('data:image'):
            return self._create_parts_from_data_uri(img_src)
        if img_src.startswith(('http://', 'https://')):
            return self._create_parts_from_url(img_src)
        if img_src.startswith('frames:'):
            return self._create_parts_from_frames(img_src)
        return self._create_parts_from_local_path(img_src)

    def _create_parts_from_data_uri(self, img_src: str) -> list[types.Part]:
        header, encoded = img_src.split(',', 1)
        mime_type = header.split(';')[0].split(':')[1]
        image_bytes = base64.b64decode(encoded)
        part = genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes))
        return [part]

    def _create_parts_from_url(self, img_src: str) -> list[types.Part]:
        response = requests.get(img_src, timeout=10)
        response.raise_for_status()
        image_bytes = response.content
        mime_type = response.headers.get('Content-Type', 'application/octet-stream')
        if not mime_type.startswith('image/'):
            UserMessage(f"URL content type '{mime_type}' does not appear to be an image for: {img_src}. Attempting to use anyway.")
        part = genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes))
        return [part]

    def _create_parts_from_frames(self, img_src: str) -> list[types.Part]:
        temp_path = img_src.replace('frames:', '')
        parts = temp_path.split(':', 1)
        if len(parts) != 2:
            UserMessage(f"Invalid 'frames:' format: {img_src}")
            return []
        max_used_frames_str, actual_path = parts
        try:
            max_used_frames = int(max_used_frames_str)
        except ValueError:
            UserMessage(f"Invalid max_frames number in 'frames:' format: {img_src}")
            return []
        frame_buffers, ext = encode_media_frames(actual_path)
        mime_type = f'image/{ext.lower()}' if ext else 'application/octet-stream'
        if ext and ext.lower() == 'jpg':
            mime_type = 'image/jpeg'
        if not frame_buffers:
            UserMessage(f"encode_media_frames returned no frames for: {actual_path}")
            return []
        step = max(1, len(frame_buffers) // 50)
        indices = list(range(0, len(frame_buffers), step))[:max_used_frames]
        parts_list: list[types.Part] = []
        for frame_idx in indices:
            if frame_idx < len(frame_buffers):
                image_bytes = frame_buffers[frame_idx]
                parts_list.append(genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes)))
        return parts_list

    def _create_parts_from_local_path(self, img_src: str) -> list[types.Part]:
        local_file_path = Path(img_src)
        if not local_file_path.is_file():
            UserMessage(f"Local image file not found: {img_src}")
            return []
        image_bytes = local_file_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(local_file_path)
        if mime_type is None:
            file_ext = local_file_path.suffix.lower().lstrip('.')
            if file_ext == 'jpg':
                mime_type = 'image/jpeg'
            elif file_ext == 'png':
                mime_type = 'image/png'
            elif file_ext == 'gif':
                mime_type = 'image/gif'
            elif file_ext == 'webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'application/octet-stream'
        part = genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes))
        return [part]

    def _handle_video_content(self, content: str):
        """Handle video content by uploading to Gemini"""
        try:
            pattern = r'<<video:(.*?):>>'
            matches = re.findall(pattern, content)
            if not matches:
                return None

            video_path = matches[0].strip()
            if video_path.startswith('http'):
                UserMessage("URL videos not yet supported for Gemini")
                return None
            # Upload local video
            return genai.upload_file(video_path)
        except Exception as e:
            UserMessage(f"Failed to process video: {e}")
            return None

    def _handle_audio_content(self, content: str):
        """Handle audio content by uploading to Gemini"""
        try:
            pattern = r'<<audio:(.*?):>>'
            matches = re.findall(pattern, content)
            if not matches:
                return None

            audio_path = matches[0].strip()
            if audio_path.startswith('http'):
                UserMessage("URL audio not yet supported for Gemini")
                return None
            # Upload local audio
            return genai.upload_file(audio_path)
        except Exception as e:
            UserMessage(f"Failed to process audio: {e}")
            return None

    def _remove_media_patterns(self, text: str) -> str:
        """Remove media pattern markers from text"""
        patterns = [
            r'<<vision:(.*?):>>',
            r'<<video:(.*?):>>',
            r'<<audio:(.*?):>>',
            r'<<document:(.*?):>>'
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text)

        return text

    def _process_multimodal_content(self, processed_input_str: str) -> list:
        """Process all multimodal content and return list of media files"""
        media_content = []

        # Process document content
        if '<<document:' in processed_input_str:
            doc_file = self._handle_document_content(processed_input_str)
            if doc_file:
                media_content.append(doc_file)

        # Process image content
        if '<<vision:' in processed_input_str:
            image_files = self._handle_image_content(processed_input_str)
            media_content.extend(image_files)

        # Process video content
        if '<<video:' in processed_input_str:
            video_file = self._handle_video_content(processed_input_str)
            if video_file:
                media_content.append(video_file)

        # Process audio content
        if '<<audio:' in processed_input_str:
            audio_file = self._handle_audio_content(processed_input_str)
            if audio_file:
                media_content.append(audio_file)

        return media_content

    def _collect_response(self, res) -> dict[str, str]:
        """Collect thinking and text content from response"""
        thinking_content = ""
        text_content = ""

        if hasattr(res, 'candidates') and res.candidates:
            candidate = res.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        if hasattr(part, 'thought') and part.thought:
                            thinking_content += part.text
                        else:
                            text_content += part.text

        return {
            "thinking": thinking_content,
            "text": text_content
        }

    def forward(self, argument):
        kwargs = argument.kwargs
        _system, prompt = argument.prop.prepared_input
        payload = self._prepare_request_payload(argument)
        except_remedy = kwargs.get('except_remedy')

        contents = self._build_contents_from_prompt(prompt)

        try:
            generation_config = self._build_generation_config(payload)
            res = self._generate_model_response(kwargs, contents, generation_config)
        except Exception as e:
            res = self._handle_generation_error(e, except_remedy, argument)

        metadata = {'raw_output': res}
        if payload.get('tools'):
            metadata = self._process_function_calls(res, metadata)

        if kwargs.get('raw_output', False):
            return [res], metadata

        output = self._collect_response(res)

        if output['thinking']:
            metadata['thinking'] = output['thinking']

        processed_text = output['text']
        if argument.prop.response_format:
            processed_text = processed_text.replace('```json', '').replace('```', '')

        return [processed_text], metadata

    def _build_contents_from_prompt(self, prompt) -> list[types.Content]:
        contents: list[types.Content] = []
        for msg in prompt:
            role = msg['role']
            parts_list = msg['content']
            contents.append(types.Content(role=role, parts=parts_list))
        return contents

    def _build_generation_config(self, payload: dict) -> types.GenerateContentConfig:
        generation_config = types.GenerateContentConfig(
            max_output_tokens=payload.get('max_output_tokens'),
            temperature=payload.get('temperature', 1.0),
            top_p=payload.get('top_p', 0.95),
            top_k=payload.get('top_k', 40),
            stop_sequences=payload.get('stop_sequences'),
            response_mime_type=payload.get('response_mime_type', 'text/plain'),
        )
        self._apply_optional_config_fields(generation_config, payload)
        return generation_config

    def _apply_optional_config_fields(self, generation_config: types.GenerateContentConfig, payload: dict) -> None:
        if payload.get('system_instruction'):
            generation_config.system_instruction = payload['system_instruction']
        if payload.get('thinking_config'):
            generation_config.thinking_config = payload['thinking_config']
        if payload.get('tools'):
            generation_config.tools = payload['tools']
            generation_config.automatic_function_calling = payload['automatic_function_calling']

    def _generate_model_response(self, kwargs: dict, contents: list[types.Content], generation_config: types.GenerateContentConfig):
        return self.client.models.generate_content(
            model=kwargs.get('model', self.model),
            contents=contents,
            config=generation_config
        )

    def _handle_generation_error(self, exception: Exception, except_remedy, argument):
        if self.api_key is None or self.api_key == '':
            msg = 'Google API key is not set. Please set it in the config file or pass it as an argument to the command method.'
            UserMessage(msg)
            if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                UserMessage(msg, raise_with=ValueError)
            self.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']
            genai.configure(api_key=self.api_key)
        if except_remedy is not None:
            return except_remedy(self, exception, self.client.generate_content, argument)
        UserMessage(f'Error during generation. Caused by: {exception}', raise_with=ValueError)
        return None

    def _process_function_calls(self, res, metadata):
        hit = False
        if hasattr(res, 'candidates') and res.candidates:
            candidate = res.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        if hit:
                            UserMessage("Multiple function calls detected in the response but only the first one will be processed.")
                            break
                        func_call = part.function_call
                        metadata['function_call'] = {
                            'name': func_call.name,
                            'arguments': func_call.args
                        }
                        hit = True
        return metadata

    def _prepare_raw_input(self, argument):
        if not argument.prop.processed_input:
            UserMessage('Need to provide a prompt instruction to the engine if `raw_input` is enabled!', raise_with=ValueError)

        raw_prompt_data = argument.prop.processed_input
        normalized_prompts = self._normalize_raw_prompt_data(raw_prompt_data)
        system_instruction, non_system_messages = self._separate_system_instruction(normalized_prompts)
        messages_for_api = self._build_raw_input_messages(non_system_messages)
        return system_instruction, messages_for_api

    def _normalize_raw_prompt_data(self, raw_prompt_data):
        if isinstance(raw_prompt_data, str):
            return [{'role': 'user', 'content': raw_prompt_data}]
        if isinstance(raw_prompt_data, dict):
            return [raw_prompt_data]
        if isinstance(raw_prompt_data, list):
            for item in raw_prompt_data:
                if not isinstance(item, dict):
                    UserMessage(f"Invalid item in raw_input list: {item}. Expected dict.", raise_with=ValueError)
            return raw_prompt_data
        UserMessage(f"Unsupported type for raw_input: {type(raw_prompt_data)}. Expected str, dict, or list of dicts.", raise_with=ValueError)
        return []

    def _separate_system_instruction(self, normalized_prompts):
        system_instruction = None
        non_system_messages = []
        for msg in normalized_prompts:
            role = msg.get('role')
            content = msg.get('content')
            if role is None or content is None:
                UserMessage(f"Message in raw_input is missing 'role' or 'content': {msg}", raise_with=ValueError)
            if not isinstance(content, str):
                UserMessage(f"Message content for role '{role}' in raw_input must be a string. Found type: {type(content)} for content: {content}", raise_with=ValueError)
            if role == 'system':
                if system_instruction is not None:
                    UserMessage('Only one system instruction is allowed in raw_input mode!', raise_with=ValueError)
                system_instruction = content
            else:
                non_system_messages.append({'role': role, 'content': content})
        return system_instruction, non_system_messages

    def _build_raw_input_messages(self, messages):
        messages_for_api = []
        for msg in messages:
            content_str = str(msg.get('content', ''))
            current_message_api_parts: list[types.Part] = []
            image_api_parts = self._handle_image_content(content_str)
            if image_api_parts:
                current_message_api_parts.extend(image_api_parts)
            text_only_content = self._remove_media_patterns(content_str)
            if text_only_content:
                current_message_api_parts.append(types.Part(text=text_only_content))
            if current_message_api_parts:
                messages_for_api.append({
                    'role': msg['role'],
                    'content': current_message_api_parts
                })
        return messages_for_api

    def prepare(self, argument):
        #@NOTE: OpenAI compatibility at high level
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        processed_input_str = str(argument.prop.processed_input)
        media_content = self._process_multimodal_content(processed_input_str)
        system_content = self._compose_system_content(argument)
        user_content = self._compose_user_content(argument)
        system_content, user_content = self._apply_self_prompt_if_needed(argument, system_content, user_content)

        user_prompt = self._build_user_prompt(media_content, user_content)
        argument.prop.prepared_input = (system_content, [user_prompt])

    def _compose_system_content(self, argument) -> str:
        system_content = ""
        _non_verbose_output = """<META_INSTRUCTION/>\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n\n"""
        if argument.prop.suppress_verbose_output:
            system_content += _non_verbose_output
        system_content = f'{system_content}\n' if system_content and len(system_content) > 0 else ''
        if argument.prop.response_format:
            response_format = argument.prop.response_format
            assert response_format.get('type') is not None, 'Response format type is required!'
            if response_format["type"] == "json_object":
                system_content += '<RESPONSE_FORMAT/>\nYou are a helpful assistant designed to output JSON.\n\n'
        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system_content += f"<STATIC_CONTEXT/>\n{static_ctxt}\n\n"
        if len(dyn_ctxt) > 0:
            system_content += f"<DYNAMIC_CONTEXT/>\n{dyn_ctxt}\n\n"
        payload = argument.prop.payload
        if argument.prop.payload:
            system_content += f"<ADDITIONAL_CONTEXT/>\n{payload!s}\n\n"
        examples: list[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system_content += f"<EXAMPLES/>\n{examples!s}\n\n"
        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            val = self._remove_media_patterns(val)
            system_content += f"<INSTRUCTION/>\n{val}\n\n"
        if argument.prop.template_suffix:
            system_content += f' You will only generate content for the placeholder `{argument.prop.template_suffix!s}` following the instructions and the provided context information.\n\n'
        return system_content

    def _compose_user_content(self, argument) -> str:
        suffix = str(argument.prop.processed_input)
        suffix = self._remove_media_patterns(suffix)
        return f"{suffix}"

    def _apply_self_prompt_if_needed(self, argument, system_content: str, user_content: str):
        if argument.prop.instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt:
            self_prompter = SelfPrompt()
            res = self_prompter(
                {'user': user_content, 'system': system_content},
                max_tokens=argument.kwargs.get('max_tokens', self.max_response_tokens),
                thinking=argument.kwargs.get('thinking', None),
            )
            if res is None:
                UserMessage("Self-prompting failed!", raise_with=ValueError)
            user_content = res['user']
            system_content = res['system']
        return system_content, user_content

    def _build_user_prompt(self, media_content, user_content: str) -> dict:
        all_user_content = list(media_content)
        if user_content.strip():
            all_user_content.append(genai.types.Part(text=user_content.strip()))
        if not all_user_content:
            all_user_content = [genai.types.Part(text="N/A")]
        return {'role': 'user', 'content': all_user_content}

    def _prepare_request_payload(self, argument):
        kwargs = argument.kwargs

        payload = {
            "max_output_tokens": kwargs.get('max_tokens', self.max_response_tokens),
            "temperature": kwargs.get('temperature', 1.0),
            "top_p": kwargs.get('top_p', 0.95),
            "top_k": kwargs.get('top_k', 40),
            "stop_sequences": kwargs.get('stop', None),
            "stream": kwargs.get('stream', False),
        }

        system, _ = argument.prop.prepared_input
        if system and system.strip():
            payload['system_instruction'] = system.strip()

        thinking_arg = kwargs.get('thinking', None)
        if thinking_arg and isinstance(thinking_arg, dict):
            thinking_budget = thinking_arg.get("thinking_budget", 1024)
            payload['thinking_config'] = types.ThinkingConfig(include_thoughts=True, thinking_budget=thinking_budget)

        response_format = kwargs.get('response_format', None)
        if response_format and response_format.get('type') == 'json_object':
            payload['response_mime_type'] = 'application/json'

        tools = kwargs.get('tools')
        if tools:
            payload['tools'] = self._convert_tools_format(tools)
            payload['automatic_function_calling'] = types.AutomaticFunctionCallingConfig(
                disable=kwargs.get('automatic_function_calling', True)
            )

        return payload

    def _convert_tools_format(self, tools):
        if tools is None:
            return None

        if not isinstance(tools, list):
            tools = [tools]

        processed_tools = []
        for tool_item in tools:
            if callable(tool_item):
                processed_tools.append(tool_item)
            elif isinstance(tool_item, types.FunctionDeclaration):
                processed_tools.append(types.Tool(function_declarations=[tool_item]))
            else:
                UserMessage(f"Ignoring invalid tool format. Expected a callable, google.genai.types.Tool, or google.genai.types.FunctionDeclaration: {tool_item}")

        if not processed_tools:
            return None

        return processed_tools
