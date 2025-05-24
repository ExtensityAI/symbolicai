import logging
import io
import re
from copy import deepcopy
import base64
import mimetypes
from pathlib import Path
import urllib.parse
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

logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


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
        self.tokenizer = None # TODO: Implement token counting for Gemini
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

    def compute_required_tokens(self, messages):
        CustomUserWarning("Token counting not implemented for Gemini", raise_with=NotImplementedError)

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
                    if hasattr(part, 'thinking') and part.thinking:
                        thinking_content += part.thinking
                    elif hasattr(part, 'text') and part.text:
                        text_content += part.text

        return {
            "thinking": thinking_content,
            "text": text_content
        }

    def forward(self, argument):
        kwargs = argument.kwargs
        contents = argument.prop.prepared_input
        payload = self._prepare_request_payload(argument)
        except_remedy = kwargs.get('except_remedy')

        try:
            generation_config = types.GenerateContentConfig(
                max_output_tokens=payload.get('max_output_tokens'),
                temperature=payload.get('temperature', 1.0),
                top_p=payload.get('top_p', 0.95),
                top_k=payload.get('top_k', 40),
                stop_sequences=payload.get('stop_sequences'),
                response_mime_type=payload.get('response_mime_type', 'text/plain'),
            )

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
        output = self._collect_response(res)

        if output['thinking']:
            metadata['thinking'] = output['thinking']

        if payload.get('tools'):
            metadata = self._process_function_calls(res, metadata)

        if argument.prop.response_format:
            # Safely remove JSON markdown formatting if present
            text_content = output['text']
            text_content = text_content.replace('```json', '').replace('```', '')
            output['text'] = text_content

        return [output['text']], metadata

    def _process_function_calls(self, res, metadata):
        # Extract function call from Gemini response and add to metadata
        if hasattr(res, 'candidates') and res.candidates:
            candidate = res.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        func_call = part.function_call
                        metadata['function_call'] = {
                            'name': func_call.name,
                            'arguments': func_call.args
                        }
                        break
        return metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            if not argument.prop.processed_input:
                CustomUserWarning('Need to provide a prompt instruction to the engine if `raw_input` is enabled!', raise_with=ValueError)

            content = argument.prop.processed_input
            if isinstance(content, str):
                content = [content]
            argument.prop.prepared_input = content
            return

        _non_verbose_output = """<META_INSTRUCTION/>\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n\n"""

        system_content = ""
        user_content = ""

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

        contents = []

        if system_content.strip():
            contents.append(system_content.strip())

        contents.extend(media_content)

        if user_content.strip():
            contents.append(user_content.strip())

        argument.prop.prepared_input = contents

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

        thinking_arg = kwargs.get('thinking', None)
        if thinking_arg and isinstance(thinking_arg, dict) and thinking_arg.get("enabled") is True:
            payload['thinking_config'] = types.ThinkingConfig(include_thoughts=True)

        response_format = kwargs.get('response_format', None)
        if response_format and response_format.get('type') == 'json_object':
            payload['response_mime_type'] = 'application/json'

        tools = kwargs.get('tools')
        if tools:
            payload['tools'] = tools
            payload['automatic_function_calling'] = types.AutomaticFunctionCallingConfig(
                disable=kwargs.get('automatic_function_calling', True)
            )

        return payload

    def _convert_tools_format(self, tools):
        if not isinstance(tools, list):
            tools = [tools]

        valid = [t for t in tools if callable(t)]
        if len(valid) < len(tools):
            CustomUserWarning("Some tools were ignored because they are not Python callables.")
        return valid

        CustomUserWarning(f"Tools argument must be a callable or list of callables, got: {tools}", raise_with=ValueError)
