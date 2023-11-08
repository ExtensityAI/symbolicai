from ...symbol import Symbol, Expression
from ...interfaces import Interface
from . import Persona
from typing import List, Tuple
import os
import logging
from pydub import AudioSegment


logger = logging.getLogger('pydub').setLevel(logging.WARNING)


class Dialogue(Expression):
    def __init__(self, *bots: Persona, n_turns: int = 10):
        assert len(bots) == 2, 'Currently dialogue requires exactly two bots.'
        self.bots = bots
        for bot in bots:
            bot.auto_print = False
        self.n_turns = n_turns
        self.value = []
        self.tts = Interface('tts')

    def print_response(self, tag: str, message: str):
        self.value.append((tag, message))
        print(f'[{tag}]: {message}\n')

    def forward(self, initial_message: str, *system_instructions: List[str]):
        starting_bot = self.bots[0]
        responding_bot = self.bots[1]
        # entangle the bots
        starting_bot.user_tag = responding_bot.bot_tag
        responding_bot.user_tag = starting_bot.bot_tag
        val = f"[SYSTEM_INSTRUCTION::]: <<<\n{starting_bot.bot_tag} talks and references -> {responding_bot.bot_tag} | {responding_bot.bot_tag} talks and references -> {starting_bot.bot_tag}."
        starting_bot.store_system_message(val)
        responding_bot.store_system_message(val)
        # set the system instructions
        if len(system_instructions) == 1:
            starting_bot.store_system_message(system_instructions[0])
            responding_bot.store_system_message(system_instructions[0])
        elif len(system_instructions) == 2:
            starting_bot.store_system_message(system_instructions[0])
            responding_bot.store_system_message(system_instructions[1])
        # initialize the conversation
        rsp1 = starting_bot.build_tag(starting_bot.bot_tag, initial_message)
        starting_bot.store(rsp1)
        self.print_response(starting_bot.bot_tag, initial_message)
        rsp1 = initial_message
        # Run the conversation for n_turns
        for i in range(self.n_turns):
            rsp2 = responding_bot(rsp1)
            self.print_response(responding_bot.bot_tag, rsp2)
            # In the subsequent turns, use the last response from the responding bot as the query for the starting bot
            if i < self.n_turns - 1:
                rsp1 = starting_bot(rsp2)
                self.print_response(starting_bot.bot_tag, rsp1)
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




