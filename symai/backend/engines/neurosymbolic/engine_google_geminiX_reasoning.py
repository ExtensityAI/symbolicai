import base64
import io
import logging
import mimetypes
import re
import urllib.parse
from copy import deepcopy
from pathlib import Path

import requests
from google import genai
from google.genai import types

from ....components import SelfPrompt
from ....misc.console import ConsoleStyle
from ....symbol import Symbol
from ....utils import CustomUserWarning, encode_media_frames
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
            if not isinstance(msg, list):
                msg = [msg]
            for part in msg:
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
            logging.error(f"Gemini count_tokens failed: {e}")
            CustomUserWarning(f"Error counting tokens for Gemini: {str(e)}", raise_with=RuntimeError)

    def compute_remaining_tokens(self, prompts: list) -> int:
        CustomUserWarning("Token counting not implemented for Gemini", raise_with=NotImplementedError)

    def _handle_document_content(self, content: str):
        """Handle document content by uploading to Gemini"""
        try:
            pattern = r'<<document:(.*?):>>'
            matches = re.findall(pattern, content)
            if not matches:
                return None

            doc_path = matches[0].strip()
            if doc_path.startswith('http'):
                CustomUserWarning("URL documents not yet supported for Gemini")
                return None
            else:
                uploaded_file = genai.upload_file(doc_path)
                return uploaded_file
        except Exception as e:
            CustomUserWarning(f"Failed to process document: {e}")
            return None

    def _handle_image_content(self, content: str) -> list:
        """Handle image content by processing and preparing google.generativeai.types.Part objects."""
        image_parts = []
        pattern = r'<<vision:(.*?):>>'
        matches = re.findall(pattern, content) # re must be imported

        for match in matches:
            img_src = match.strip()

            try:
                if img_src.startswith('data:image'):
                    header, encoded = img_src.split(',', 1)
                    mime_type = header.split(';')[0].split(':')[1]
                    image_bytes = base64.b64decode(encoded)
                    image_parts.append(genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes)))

                elif img_src.startswith('http://') or img_src.startswith('https://'):
                    response = requests.get(img_src, timeout=10) # 10 seconds timeout
                    response.raise_for_status()

                    image_bytes = response.content
                    mime_type = response.headers.get('Content-Type', 'application/octet-stream')

                    if not mime_type.startswith('image/'):
                        CustomUserWarning(f"URL content type '{mime_type}' does not appear to be an image for: {img_src}. Attempting to use anyway.")

                    image_parts.append(genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes)))

                elif img_src.startswith('frames:'):
                    temp_path = img_src.replace('frames:', '')
                    parts = temp_path.split(':', 1)
                    if len(parts) != 2:
                        CustomUserWarning(f"Invalid 'frames:' format: {img_src}")
                        continue
                    max_used_frames_str, actual_path = parts
                    try:
                        max_used_frames = int(max_used_frames_str)
                    except ValueError:
                        CustomUserWarning(f"Invalid max_frames number in 'frames:' format: {img_src}")
                        continue

                    frame_buffers, ext = encode_media_frames(actual_path)

                    mime_type = f'image/{ext.lower()}' if ext else 'application/octet-stream'
                    if ext and ext.lower() == 'jpg':
                        mime_type = 'image/jpeg'

                    if not frame_buffers:
                        CustomUserWarning(f"encode_media_frames returned no frames for: {actual_path}")
                        continue

                    step = max(1, len(frame_buffers) // 50)
                    indices = list(range(0, len(frame_buffers), step))[:max_used_frames]

                    for i_idx in indices:
                        if i_idx < len(frame_buffers):
                           image_bytes = frame_buffers[i_idx]
                           image_parts.append(genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes)))

                else:
                    # Handle local file paths
                    local_file_path = Path(img_src)
                    if not local_file_path.is_file():
                        CustomUserWarning(f"Local image file not found: {img_src}")
                        continue

                    image_bytes = local_file_path.read_bytes()
                    mime_type, _ = mimetypes.guess_type(local_file_path)
                    if mime_type is None: # Fallback MIME type determination
                        file_ext = local_file_path.suffix.lower().lstrip('.')
                        if file_ext == 'jpg': mime_type = 'image/jpeg'
                        elif file_ext == 'png': mime_type = 'image/png'
                        elif file_ext == 'gif': mime_type = 'image/gif'
                        elif file_ext == 'webp': mime_type = 'image/webp'
                        else: mime_type = 'application/octet-stream'

                    image_parts.append(genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes)))

            except Exception as e:
                CustomUserWarning(f"Failed to process image source '{img_src}'. Error: {str(e)}", raise_with=ValueError)

        return image_parts

    def _handle_video_content(self, content: str):
        """Handle video content by uploading to Gemini"""
        try:
            pattern = r'<<video:(.*?):>>'
            matches = re.findall(pattern, content)
            if not matches:
                return None

            video_path = matches[0].strip()
            if video_path.startswith('http'):
                CustomUserWarning("URL videos not yet supported for Gemini")
                return None
            else:
                # Upload local video
                uploaded_file = genai.upload_file(video_path)
                return uploaded_file
        except Exception as e:
            CustomUserWarning(f"Failed to process video: {e}")
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
                CustomUserWarning("URL audio not yet supported for Gemini")
                return None
            else:
                # Upload local audio
                uploaded_file = genai.upload_file(audio_path)
                return uploaded_file
        except Exception as e:
            CustomUserWarning(f"Failed to process audio: {e}")
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
        system, prompt = argument.prop.prepared_input
        payload = self._prepare_request_payload(argument)
        except_remedy = kwargs.get('except_remedy')

        contents = []
        for msg in prompt:
            role = msg['role']
            parts_list = msg['content']
            contents.append(types.Content(role=role, parts=parts_list))

        try:
            generation_config = types.GenerateContentConfig(
                max_output_tokens=payload.get('max_output_tokens'),
                temperature=payload.get('temperature', 1.0),
                top_p=payload.get('top_p', 0.95),
                top_k=payload.get('top_k', 40),
                stop_sequences=payload.get('stop_sequences'),
                response_mime_type=payload.get('response_mime_type', 'text/plain'),
            )

            if payload.get('system_instruction'):
                generation_config.system_instruction = payload['system_instruction']

            if payload.get('thinking_config'):
                generation_config.thinking_config = payload['thinking_config']

            if payload.get('tools'):
                generation_config.tools = payload['tools']
                generation_config.automatic_function_calling=payload['automatic_function_calling']

            res = self.client.models.generate_content(
                model=kwargs.get('model', self.model),
                contents=contents,
                config=generation_config
            )

        except Exception as e:
            if self.api_key is None or self.api_key == '':
                msg = 'Google API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                logging.error(msg)
                if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                    CustomUserWarning(msg, raise_with=ValueError)
                self.api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']
                genai.configure(api_key=self.api_key)

            if except_remedy is not None:
                res = except_remedy(self, e, self.client.generate_content, argument)
            else:
                CustomUserWarning(f"Error during generation: {str(e)}", raise_with=ValueError)

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
            # Safely remove JSON markdown formatting if present
            processed_text = processed_text.replace('```json', '').replace('```', '')

        return [processed_text], metadata

    def _process_function_calls(self, res, metadata):
        hit = False
        if hasattr(res, 'candidates') and res.candidates:
            candidate = res.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        if hit:
                            CustomUserWarning("Multiple function calls detected in the response but only the first one will be processed.")
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
            CustomUserWarning('Need to provide a prompt instruction to the engine if `raw_input` is enabled!', raise_with=ValueError)

        raw_prompt_data = argument.prop.processed_input
        messages_for_api = []
        system_instruction = None

        if isinstance(raw_prompt_data, str):
            normalized_prompts = [{'role': 'user', 'content': raw_prompt_data}]
        elif isinstance(raw_prompt_data, dict):
            normalized_prompts = [raw_prompt_data]
        elif isinstance(raw_prompt_data, list):
            for item in raw_prompt_data:
                if not isinstance(item, dict):
                    CustomUserWarning(f"Invalid item in raw_input list: {item}. Expected dict.", raise_with=ValueError)
            normalized_prompts = raw_prompt_data
        else:
            CustomUserWarning(f"Unsupported type for raw_input: {type(raw_prompt_data)}. Expected str, dict, or list of dicts.", raise_with=ValueError)

        temp_non_system_messages = []
        for msg in normalized_prompts:
            role = msg.get('role')
            content = msg.get('content')

            if role is None or content is None:
                CustomUserWarning(f"Message in raw_input is missing 'role' or 'content': {msg}", raise_with=ValueError)
            if not isinstance(content, str):
                CustomUserWarning(f"Message content for role '{role}' in raw_input must be a string. Found type: {type(content)} for content: {content}", raise_with=ValueError)
            if role == 'system':
                if system_instruction is not None:
                    CustomUserWarning('Only one system instruction is allowed in raw_input mode!', raise_with=ValueError)
                system_instruction = content
            else:
                temp_non_system_messages.append({'role': role, 'content': content})

        for msg in temp_non_system_messages:
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

        return system_instruction, messages_for_api

    def prepare(self, argument):
        #@NOTE: OpenAI compatibility at high level
        if argument.prop.raw_input:
            argument.prop.prepared_input = self._prepare_raw_input(argument)
            return

        _non_verbose_output = """<META_INSTRUCTION/>\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n\n"""

        user_content = ""
        system_content = ""

        if argument.prop.suppress_verbose_output:
            system_content += _non_verbose_output
        system_content = f'{system_content}\n' if system_content and len(system_content) > 0 else ''

        if argument.prop.response_format:
            _rsp_fmt = argument.prop.response_format
            assert _rsp_fmt.get('type') is not None, 'Response format type is required!'
            if _rsp_fmt["type"] == "json_object":
                system_content += f'<RESPONSE_FORMAT/>\nYou are a helpful assistant designed to output JSON.\n\n'

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system_content += f"<STATIC_CONTEXT/>\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system_content += f"<DYNAMIC_CONTEXT/>\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system_content += f"<ADDITIONAL_CONTEXT/>\n{str(payload)}\n\n"

        examples: list[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system_content += f"<EXAMPLES/>\n{str(examples)}\n\n"

        # Handle multimodal content
        processed_input_str = str(argument.prop.processed_input)
        media_content = self._process_multimodal_content(processed_input_str)

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            val = self._remove_media_patterns(val)
            system_content += f"<INSTRUCTION/>\n{val}\n\n"

        suffix = str(argument.prop.processed_input)
        suffix = self._remove_media_patterns(suffix)
        user_content += f"{suffix}"

        if argument.prop.template_suffix:
            system_content += f' You will only generate content for the placeholder `{str(argument.prop.template_suffix)}` following the instructions and the provided context information.\n\n'

        # Handle self-prompting
        if argument.prop.instance._kwargs.get('self_prompt', False) or argument.prop.self_prompt:
            self_prompter = SelfPrompt()

            res = self_prompter(
                {'user': user_content, 'system': system_content},
                max_tokens=argument.kwargs.get('max_tokens', self.max_response_tokens),
                thinking=argument.kwargs.get('thinking', None),
            )
            if res is None:
                CustomUserWarning("Self-prompting failed!", raise_with=ValueError)

            user_content = res['user']
            system_content = res['system']

        all_user_content = []
        all_user_content.extend(media_content) #
        if user_content.strip():
            all_user_content.append(genai.types.Part(text=user_content.strip()))

        if not all_user_content:
            all_user_content = [genai.types.Part(text="N/A")]

        user_prompt = {'role': 'user', 'content': all_user_content}

        argument.prop.prepared_input = (system_content, [user_prompt])

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
                CustomUserWarning(f"Ignoring invalid tool format. Expected a callable, google.genai.types.Tool, or google.genai.types.FunctionDeclaration: {tool_item}")

        if not processed_tools:
            return None

        return processed_tools
