import sys
import json
import webbrowser

from pathlib import Path
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.shortcuts import yes_no_dialog, input_dialog, button_dialog

from ..misc.console import ConsoleStyle


def show_splash_screen(print: callable = print_formatted_text):
    print('\n\n')
    print('- '*42)
    print('=='*41 + '=')
    print(r'''
      ____|        |                      _)  |              \    _ _|
      __|  \ \  /  __|   _ \  __ \    __|  |  __|  |   |    _ \     |
      |     `  <   |     __/  |   | \__ \  |  |    |   |   ___ \    |
     _____| _/\_\ \__| \___| _|  _| ____/ _| \__| \__, | _/    _\ ___|
                                                  ____/
    ''', escape=True)
    print('- '*17 + ' ExtensityAI ' + ' -'*18  + '\n')


def show_info_message(print: callable = print_formatted_text):
    print('Welcome to SymbolicAI!' + '\n')
    print('SymbolicAI is an open-source Python project for building AI-powered applications\nand assistants.')
    print('We utilize the power of large language models and the latest research in AI.' + '\n')
    print('SymbolicAI is backed by ExtensityAI. We are committed to open research,\nthe democratization of AI tools and much more ...' + '\n')

    print('... and we also like peanut butter and jelly sandwiches, and cookies.' + '\n\n')
    print('If you like what we are doing please help us achieve our mission!')
    print('More information is available at https://www.extensity.ai' + '\n')


def show_separator(print: callable = print_formatted_text):
    print('- '*42 + '\n')


def is_openai_api_model(key: str):
    if key is None:
        return False
    return 'gpt' in key or 'text' in key


def show_main_setup_menu(show_wizard: bool = True):
    # Step 0: Load config
    symai_config_path = Path.cwd() / 'symai.config.json'
    if not symai_config_path.exists():
        root_dir  = Path.home() / '.symai'
        symai_config_path = root_dir / 'symai.config.json'

    if symai_config_path.exists():
        with open(symai_config_path, 'r', encoding="utf-8") as f:
            SYMAI_CONFIG = json.load(f)
    else:
        SYMAI_CONFIG = {}

    root_package = Path(__file__).parent.parent
    terms_of_services = root_package / 'TERMS_OF_SERVICE.md'
    with open(terms_of_services, 'r', encoding="utf-8") as f:
        TERMS_OF_SERVICES = f.read()

    # define defaults variables
    agreed                          = False
    support_community               = False
    donation_result                 = False
    nesy_engine_model               = "gpt-3.5-turbo"
    nesy_engine_api_key             = ''
    embedding_engine_api_key        = ''
    embedding_model                 = "text-embedding-ada-002"
    symbolic_engine_api_key         = ''
    symbolic_engine_model           = "wolframalpha"
    imagerendering_engine_api_key   = ''
    vision_engine_model             = "openai/clip-vit-base-patch32"
    search_engine_api_key           = ''
    search_engine_model             = "google"
    ocr_engine_api_key              = ''
    speech_to_text_engine_model     = "base"
    text_to_speech_engine_model     = "tts-1"
    text_to_speech_engine_voice     = "echo"
    indexing_engine_api_key         = ''
    indexing_engine_environment     = "us-west1-gcp"
    caption_engine_environment      = "blip2_opt/pretrain_opt2.7b"
    text_to_speech_engine_api_key   = ''

    if show_wizard:
        try:
            # Step 1: Accept terms and services
            agreed = yes_no_dialog(
                title="Terms of Service",
                text=f"Do you accept the terms of service and privacy policy?\n\nFILE:{terms_of_services}\n{TERMS_OF_SERVICES}",
            ).run()
            if not agreed:
                with ConsoleStyle('error') as console:
                    console.print("You need to accept the terms of services to continue.")
                    sys.exit(0)

            # Step 2: Enter OpenAI Key

            # Step 2.1: Select model
            nesy_engine_model = input_dialog(
                title="Select Model",
                text="Please select a OpenAI model (https://platform.openai.com/docs/models) or a custom neuro-symbolic model:",
                default=nesy_engine_model if 'NEUROSYMBOLIC_ENGINE_MODEL' not in SYMAI_CONFIG else SYMAI_CONFIG['NEUROSYMBOLIC_ENGINE_MODEL'],
            ).run()

            # Step 2.2: Enter API key
            if is_openai_api_model(nesy_engine_model):
                nesy_engine_api_key = input_dialog(
                    title="OpenAI API Key",
                    text="Please enter your OpenAI API Key (https://platform.openai.com/api-keys):",
                    default=nesy_engine_api_key if 'NEUROSYMBOLIC_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['NEUROSYMBOLIC_ENGINE_API_KEY'],
                ).run()
                if not nesy_engine_api_key:
                    with ConsoleStyle('warn') as console:
                        console.print("No API key or custom model ID provided. The framework will not work without it.")
            else:
                nesy_engine_api_key = input_dialog(
                    title="[Optional] Neuro-Symbolic Model API Key",
                    text="Please enter your Neuro-Symbolic Model API Key if applicable:",
                    default=nesy_engine_api_key if 'NEUROSYMBOLIC_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['NEUROSYMBOLIC_ENGINE_API_KEY'],
                ).run()

            # Step 2.3: Enter Embedding Engine API Key is_openai_api_model(nesy_engine_model)
            if is_openai_api_model(nesy_engine_model):
                embedding_engine_api_key = nesy_engine_api_key
            else:
                embedding_engine_api_key = input_dialog(
                    title="[Optional] Embedding Engine API Key",
                    text="Please enter your Embedding Engine API Key if applicable:",
                    default=embedding_engine_api_key if 'EMBEDDING_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['EMBEDDING_ENGINE_API_KEY'],
                ).run()

            # Ask for optional settings yes/no
            continue_optional_settings = yes_no_dialog(
                title="Optional Settings",
                text="Do you want to configure optional settings?"
            ).run()

            if nesy_engine_model is None:
                nesy_engine_model           = ''

            if continue_optional_settings:
                # Step 2.4: Enter Embedding Model
                embedding_model = input_dialog(
                    title="[Optional] Embedding Model",
                    text="Please enter an embedding model if applicable. We currently support OpenAI's embedding models:",
                    default=embedding_model if 'EMBEDDING_ENGINE_MODEL' not in SYMAI_CONFIG else SYMAI_CONFIG['EMBEDDING_ENGINE_MODEL'],
                ).run()

                # Step 2.5: Symbolic Engine API Key
                symbolic_engine_api_key = input_dialog(
                    title="[Optional] Symbolic Engine API Key",
                    text="Please enter your Symbolic Engine API Key if applicable. We currently support Wolfram Alpha as an symbolic solver. Get a Key here https://products.wolframalpha.com/api:",
                    default=symbolic_engine_api_key if 'SYMBOLIC_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['SYMBOLIC_ENGINE_API_KEY'],
                ).run()

                if symbolic_engine_api_key:
                    # Step 2.6: Symbolic Engine Model
                    symbolic_engine_model = symbolic_engine_model

                # Step 2.7: Enter Imagerendering engine api key
                if is_openai_api_model(nesy_engine_model):
                    imagerendering_engine_api_key = input_dialog(
                        title="[Optional] Image Rendering Engine API Key",
                        text="Please enter your Image Rendering Engine API Key if applicable. We currently support OpenAI's DALL-E model:",
                        default=imagerendering_engine_api_key if 'IMAGERENDERING_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['IMAGERENDERING_ENGINE_API_KEY'],
                    ).run()

                # Step 2.8: Enter Vision Engine Model
                vision_engine_model = input_dialog(
                    title="[Optional] Vision Engine Model",
                    text="Please enter your CLIP Engine Model based on HuggingFace https://huggingface.co/models?other=clip if applicable:",
                    default=vision_engine_model if 'VISION_ENGINE_MODEL' not in SYMAI_CONFIG else SYMAI_CONFIG['VISION_ENGINE_MODEL'],
                ).run()

                # Step 2.9: Enter Search Engine API Key
                search_engine_api_key = input_dialog(
                    title="[Optional] Search Engine API Key",
                    text="Please enter your Search Engine API Key if applicable. We currently support SerpApi https://serpapi.com/search-api:",
                    default=search_engine_api_key if 'SEARCH_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['SEARCH_ENGINE_API_KEY'],
                ).run()

                if search_engine_api_key:
                    # Step 2.10: Enter Search Engine Model
                    search_engine_model = input_dialog(
                        title="[Optional] Search Engine Model",
                        text="Please enter your Search Engine Model if applicable:",
                        default=search_engine_model if 'SEARCH_ENGINE_MODEL' not in SYMAI_CONFIG else SYMAI_CONFIG['SEARCH_ENGINE_MODEL'],
                    ).run()

                # Step 2.11: Enter OCR Engine API Key
                ocr_engine_api_key = input_dialog(
                    title="[Optional] OCR Engine API Key",
                    text="Please enter your OCR Engine API Key if applicable. We currently support ApiLayer https://apilayer.com/marketplace/image_to_text-api:",
                    default=ocr_engine_api_key if 'OCR_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['OCR_ENGINE_API_KEY'],
                ).run()

                # Step 2.12: Enter Speech-to-Text Engine Model
                speech_to_text_engine_model = input_dialog(
                    title="[Optional] Speech-to-Text Engine Model",
                    text="Please enter your Speech-to-Text Engine Model if applicable. We currently use whisper self-hosted models:",
                    default=speech_to_text_engine_model if 'SPEECH_TO_TEXT_ENGINE_MODEL' not in SYMAI_CONFIG else SYMAI_CONFIG['SPEECH_TO_TEXT_ENGINE_MODEL'],
                ).run()

                # Step 2.13: Enter Text-to-Speech Engine API Key
                text_to_speech_engine_api_key = input_dialog(
                    title="[Optional] Text-to-Speech Engine API Key",
                    text="Please enter your Text-to-Speech Engine API Key if applicable. Currently this is based on OpenAI's tts models:",
                    default=text_to_speech_engine_api_key if 'TEXT_TO_SPEECH_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['TEXT_TO_SPEECH_ENGINE_API_KEY'],
                ).run()

                # Step 2.14: Enter Text-to-Speech Engine Model
                text_to_speech_engine_model = input_dialog(
                    title="[Optional] Text-to-Speech Engine Model",
                    text="Please enter your Text-to-Speech Engine Model if applicable:",
                    default=text_to_speech_engine_model if 'TEXT_TO_SPEECH_ENGINE_MODEL' not in SYMAI_CONFIG else SYMAI_CONFIG['TEXT_TO_SPEECH_ENGINE_MODEL'],
                ).run()

                # Step 2.15: Enter Text-to-Speech Engine Voice
                text_to_speech_engine_voice = input_dialog(
                    title="[Optional] Text-to-Speech Engine Voice",
                    text="Please enter your Text-to-Speech Engine Voice if applicable:",
                    default=text_to_speech_engine_voice if 'TEXT_TO_SPEECH_ENGINE_VOICE' not in SYMAI_CONFIG else SYMAI_CONFIG['TEXT_TO_SPEECH_ENGINE_VOICE'],
                ).run()

                # Step 2.16: Enter Indexing Engine API Key
                indexing_engine_api_key = input_dialog(
                    title="[Optional] Indexing Engine API Key",
                    text="Please enter your Indexing Engine API Key if applicable. Currently we support Pinecone https://www.pinecone.io/product/:",
                    default=indexing_engine_api_key if 'INDEXING_ENGINE_API_KEY' not in SYMAI_CONFIG else SYMAI_CONFIG['INDEXING_ENGINE_API_KEY'],
                ).run()

                # Step 2.17: Enter Indexing Engine Environment
                indexing_engine_environment = ''
                if indexing_engine_api_key:
                    indexing_engine_environment = input_dialog(
                        title="[Optional] Indexing Engine Environment",
                        text="Please enter your Indexing Engine Environment if applicable:",
                        default=indexing_engine_environment if 'INDEXING_ENGINE_ENVIRONMENT' not in SYMAI_CONFIG else SYMAI_CONFIG['INDEXING_ENGINE_ENVIRONMENT'],
                    ).run()

                # Step 2.18: Enter Caption Engine Environment
                caption_engine_environment = input_dialog(
                    title="[Optional] Caption Engine Environment",
                    text="Please enter your Caption Engine Environment if applicable. The current implementation is based on HuggingFace https://huggingface.co/models?pipeline_tag=image-to-text:",
                    default=caption_engine_environment if 'CAPTION_ENGINE_ENVIRONMENT' not in SYMAI_CONFIG else SYMAI_CONFIG['CAPTION_ENGINE_ENVIRONMENT'],
                ).run()

            # Step 3: Enable/Disable community support and data sharing
            support_community = yes_no_dialog(
                title="Community Support and Data Sharing",
                text="Enable community support and data sharing?"
            ).run()
            if not support_community:
                with ConsoleStyle('info') as console:
                    msg = 'To support us improve our framework consider enabling this setting in the future. By doing so you  not only improve your own user experience but help us deliver new and exciting solutions in the future. Your data is uploaded to our research servers, and helps us develop on-premise solutions and the overall SymbolicAI experience. We do not sell or monetize your data otherwise. We thank you very much for supporting the research community and helping us thrive together! If you wish to update this option go to your .symai config situated in your home directory or set the environment variable `SUPPORT_COMMUNITY` to `True`.'
                    console.print(msg)

            # Step 4: Donate to the open-source collective
            donation_result = button_dialog(
                title="Support ExtensityAI @ Open-Source Collective",
                text="Would you like to donate to ExtensityAI at the open-source collective?",
                buttons=[
                    ('Yes', True),
                    ('No', False)
                ],
            ).run()
            if donation_result:
                # open link to the open collective donation page in the browser
                webbrowser.open('https://opencollective.com/symbolicai')
                with ConsoleStyle('success') as console:
                    msg = 'We are so happy that you consider to support us! Your contribution helps feed our starving cookie monster researcher staff. If you wish to further support us in the future, you can visit us at our open collective page https://opencollective.com/symbolicai and keep track of our expenses and donations. We thank you very much for your support!'
                    console.print(msg)
            else:
                with ConsoleStyle('alert') as console:
                    msg = 'We rely on donations to keep our servers running and to support our researchers. If you wish that we can thrive together please consider donating to our open collective. Without your support we cannot continue our mission, since also peanuts and jelly sandwiches are not free, or even cookies for that matter. You can donate at https://opencollective.com/symbolicai and keep track of our expenses and donations. We thank you very much for your support!'
                    console.print(msg)
        except Exception as e:
            # wizzard not supported
            with ConsoleStyle('warn') as console:
                console.print(f"Created a new config file, however, your application initialization did not finish properly and may not work as expected. Error: {e}")

    # Process the setup results
    settings = {
        'terms_agreed':                     agreed,
        'nesy_engine_model':                nesy_engine_model if nesy_engine_model else '',
        'nesy_engine_api_key':              nesy_engine_api_key,
        'symbolic_engine_api_key':          symbolic_engine_api_key,
        'symbolic_engine_model':            symbolic_engine_model,
        'support_community':                support_community,
        'donated':                          donation_result,
        'embedding_engine_api_key':         embedding_engine_api_key if embedding_engine_api_key else '',
        'embedding_model':                  embedding_model,
        'imagerendering_engine_api_key':    imagerendering_engine_api_key if imagerendering_engine_api_key else '',
        'vision_engine_model':              vision_engine_model,
        'search_engine_api_key':            search_engine_api_key,
        'search_engine_model':              search_engine_model,
        'ocr_engine_api_key':               ocr_engine_api_key,
        'text_to_speech_engine_api_key':    text_to_speech_engine_api_key,
        'speech_to_text_engine_model':      speech_to_text_engine_model,
        'text_to_speech_engine_model':      text_to_speech_engine_model,
        'text_to_speech_engine_voice':      text_to_speech_engine_voice,
        'indexing_engine_api_key':          indexing_engine_api_key,
        'indexing_engine_environment':      indexing_engine_environment,
        'caption_engine_environment':       caption_engine_environment
    }

    return settings


def show_intro_menu():
    with ConsoleStyle('extensity') as console:
        show_splash_screen(print=console.print)
    with ConsoleStyle('text') as console:
        show_info_message(print=console.print)
    with ConsoleStyle('extensity') as console:
        show_separator(print=console.print)


def show_menu(show_wizard: bool = True):
    show_intro_menu()
    return show_main_setup_menu(show_wizard=show_wizard)


if __name__ == '__main__':
    show_menu()
