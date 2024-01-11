import os
import logging

from typing import List, Tuple
from pydub import AudioSegment

from ...symbol import Expression
from ...interfaces import Interface
from . import Persona


logger = logging.getLogger('pydub').setLevel(logging.WARNING)


class Dialogue(Expression):
    def __init__(self, *bots: Persona, n_turns: int = 10, **kwargs):
        super().__init__(**kwargs)
        assert len(bots) == 2, 'Currently dialogue requires exactly two bots.'
        self.bots = bots
        for bot in bots:
            bot.auto_print = False
        self.n_turns = n_turns
        self._value = []
        self.tts = Interface('tts')

    def print_response(self, tag: str, message: str):
        self._value.append((tag, message))
        print(f'[{tag}]: {message}\n')

    def forward(self, initial_message: str, *system_instructions: List[str]):
        # Assign the two Persona objects to variables for convenience
        bot_1 = self.bots[0]
        bot_2 = self.bots[1]
        # Set the user_tag of each bot to the bot_tag of the other bot
        bot_1.user_tag = bot_2.bot_tag
        bot_2.user_tag = bot_1.bot_tag
        # Generate the system message that instructs each bot on addressing the other
        system_ = f"{bot_1.bot_tag[:-2]} only names and references -> {bot_2.bot_tag[:-2]}, " \
                f"and {bot_2.bot_tag[:-2]} only names and references -> {bot_1.bot_tag[:-2]}"
        # Store system instructions, if present
        if len(system_instructions) == 1:
            bot_1.store_system_message(system_instructions[0])
            bot_2.store_system_message(system_instructions[0])
        elif len(system_instructions) == 2:
            bot_1.store_system_message(system_instructions[0])
            bot_2.store_system_message(system_instructions[1])
        # Store the generated system message about naming and referencing
        bot_1.store_system_message(system_)
        bot_2.store_system_message(system_)
        # Must manually initialize the conversation for the first bot since the initial_message is not self generated
        tagged_message = bot_1.build_tag(bot_1.bot_tag, initial_message) # using helper function
        bot_1.store(tagged_message) # add to bot's memory
        # Print initial message
        self.print_response(bot_1.bot_tag, initial_message)
        conversation_history = [initial_message]  # Keep track of the full conversation history without tags
        # Engage in the dialogue for the specified number of turns
        for turn in range(self.n_turns):
            starting_bot, responding_bot = bot_1, bot_2
            # Get the last message from the conversation history for the starting bot to respond to
            last_message = conversation_history[-1]
            # Starting bot generates a response
            response = responding_bot.forward(last_message)
            # Save starting bot's response to conversation history
            conversation_history.append(response)
            # Print starting bot's response
            self.print_response(responding_bot.bot_tag, response)
            # Check if conversation should continue
            if turn < self.n_turns - 1:
                # Responding bot generates a response to the starting bot's message
                response = starting_bot.forward(response)
                # Save responding bot's response to conversation history
                conversation_history.append(response)
                # Print responding bot's response
                self.print_response(starting_bot.bot_tag, response)
        return self.value

    def get(self, convo: Persona) -> List[Tuple[str, str]]:
        memory = convo._memory.split(convo.marker)
        bot = []
        user = []
        for entry in memory:
            if entry.strip() == '':
                continue
            if convo.bot_tag in entry:
                bot_msg = entry.split('<<<')[-1].split('>>>')[0]
                bot.append((convo.bot_tag, bot_msg))
            elif convo.user_tag in entry:
                user_msg = entry.split('<<<')[-1].split('>>>')[0]
                user.append((convo.user_tag, user_msg))
        return bot, user

    def render(self, path, voices: List[str] = ['onyx', 'echo'], overwrite: bool = True,  combine: bool = True):
        # create temporary subfolder with all persona conversation fragments
        bot1_memory = self.get(self.bots[0])[0]
        bot2_memory = self.get(self.bots[1])[0]
        frag_path = f'{path}/tmp'
        if overwrite or not os.path.exists(frag_path):
            os.makedirs(frag_path, exist_ok=True)
            # iterate over each reply and create a new file for the fragment
            for i, (tag, msg) in enumerate(bot1_memory):
                self.tts(msg, path=os.path.join(frag_path, f'bot1_tts_{i}.mp3'), voice=voices[0])
            for i, (tag, msg) in enumerate(bot2_memory):
                self.tts(msg, path=os.path.join(frag_path, f'bot2_tts_{i}.mp3'), voice=voices[1])
        if not combine:
            return frag_path, None
        # combine the audio files into one audio file by alternating between the two bots
        combined = AudioSegment.from_file(os.path.join(frag_path, 'bot1_tts_0.mp3'), format="mp3")
        # iterate over n + 1 files, where n is the number of bot1 replies
        # read os files with ending .mp3
        files = [f for f in os.listdir(frag_path) if os.path.isfile(os.path.join(frag_path, f)) and f.endswith('.mp3')]
        for i in range(len(files)//2):
            if i > 0:
                combined += AudioSegment.from_file(os.path.join(frag_path, f'bot1_tts_{i}.mp3'), format="mp3")
            combined += AudioSegment.from_file(os.path.join(frag_path, f'bot2_tts_{i}.mp3'), format="mp3")
        export_file = os.path.join(frag_path, "output.mp3")
        file_handle = combined.export(export_file, format="mp3")
        return export_file, file_handle




