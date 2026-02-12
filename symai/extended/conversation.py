import pickle
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from ..components import FileReader
from ..interfaces import Interface
from ..memory import SlidingWindowStringConcatMemory
from ..symbol import Symbol
from ..utils import UserMessage
from .seo_query_optimizer import SEOQueryOptimizer


class CodeFormatter:
    def __call__(self, value: str, *_args: Any, **_kwds: Any) -> Any:
        # extract code from chat conversations or ```<language>\n{code}\n``` blocks
        return Symbol(value).extract(
            "Only extract code without ``` block markers or chat conversations"
        )


class Conversation(SlidingWindowStringConcatMemory):
    def __init__(
        self,
        init: str | None = None,
        file_link: list[str] | None = None,
        url_link: list[str] | None = None,
        index_name: str | None = None,
        auto_print: bool = True,
        truncation_percentage: float = 0.8,
        truncation_type: str = "head",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.truncation_percentage = truncation_percentage
        self.truncation_type = truncation_type
        self.auto_print = auto_print
        if file_link and isinstance(file_link, str):
            file_link = [file_link]
        if url_link and isinstance(url_link, str):
            url_link = [url_link]
        self.file_link = file_link
        self.url_link = url_link
        self.index_name = index_name
        self.seo_opt = SEOQueryOptimizer()
        self.reader = FileReader()
        self.scraper = Interface("naive_scrape")
        self.user_tag = "USER::"
        self.bot_tag = "ASSISTANT::"

        if init is not None:
            self.store_system_message(init, *args, **kwargs)
        if file_link is not None:
            for fl in file_link:
                self.store_file(fl, *args, **kwargs)
        if url_link is not None:
            for url in url_link:
                self.store_url(url, *args, **kwargs)
        self.indexer = None
        self.index = None
        if index_name is not None:
            UserMessage(
                "Index not supported for conversation class.", raise_with=NotImplementedError
            )

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.pop("seo_opt", None)
        state.pop("indexer", None)
        state.pop("index", None)
        state.pop("reader", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.seo_opt = SEOQueryOptimizer()
        self.reader = FileReader()
        if self.index_name is not None:
            UserMessage(
                "Index not supported for conversation class.", raise_with=NotImplementedError
            )

    def store_system_message(self, message: str, *_args, **_kwargs):
        val = f"[SYSTEM_INSTRUCTION::]: <<<\n{message!s}\n>>>\n"
        self.store(val)

    def store_file(self, file_path: str, *_args, **_kwargs):
        content = self.reader(file_path)
        val = f"[DATA::{file_path}]: <<<\n{content!s}\n>>>\n"
        self.store(val)

    def store_url(self, url: str, *_args, **_kwargs):
        content = self.scraper(url)
        val = f"[DATA::{url}]: <<<\n{content!s}\n>>>\n"
        self.store(val)

    @staticmethod
    def save_conversation_state(conversation: "Conversation", file_path: str) -> None:
        # Check if path exists and create it if it doesn't
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        # Save the conversation object as a pickle file
        with path_obj.open("wb") as handle:
            pickle.dump(conversation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_conversation_state(self, path: str) -> "Conversation":
        # Check if the file exists and it's not empty
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.stat().st_size <= 0:
                UserMessage("File is empty.", raise_with=Exception)
            # Load the conversation object from a pickle file
            with path_obj.open("rb") as handle:
                conversation_state = pickle.load(handle)
        else:
            UserMessage("File does not exist or is empty.", raise_with=Exception)

        # Create a new instance of the `Conversation` class and restore
        # the state from the saved conversation
        return self.restore(conversation_state)

    def restore(self, conversation_state: "Conversation") -> "Conversation":
        self._memory = conversation_state._memory
        self.truncation_percentage = conversation_state.truncation_percentage
        self.truncation_type = conversation_state.truncation_type
        self.auto_print = conversation_state.auto_print
        self.file_link = conversation_state.file_link
        self.url_link = conversation_state.url_link
        self.index_name = conversation_state.index_name
        self.seo_opt = SEOQueryOptimizer()
        self.reader = FileReader()
        if self.index_name is not None:
            UserMessage(
                "Index not supported for conversation class.", raise_with=NotImplementedError
            )
        return self

    def commit(self, target_file: str | None = None, formatter: Callable | None = None):
        if target_file and isinstance(target_file, str):
            file_link = target_file
        else:
            file_link = self.file_link if self.file_link else None
            if isinstance(file_link, str):
                file_link = [file_link]
            elif isinstance(file_link, list) and len(file_link) == 1:
                file_link = file_link[0]
            else:
                file_link = None  # cannot commit to multiple files
                UserMessage("Cannot commit to multiple files.", raise_with=Exception)
        if file_link:
            # if file extension is .py, then format code
            format_ = formatter
            formatter = (
                CodeFormatter() if format_ is None and file_link.endswith(".py") else formatter
            )
            val = self.value
            if formatter:
                val = formatter(val)
            # if file does not exist, create it
            with Path(file_link).open("w") as file:
                file.write(str(val))
        else:
            UserMessage("File link is not set or a set of files.", raise_with=Exception)

    def save(self, path: str, replace: bool = False) -> Symbol:
        return Symbol(self._memory).save(path, replace=replace)

    def build_tag(self, tag: str, query: str) -> str:
        # get timestamp in string format
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        return str(f"[{tag}{timestamp}]: <<<\n{query!s}\n>>>\n")

    def forward(self, query: str, *args, **kwargs):
        kwargs = self._apply_truncation_overrides(kwargs)
        query = self._to_symbol(query)
        memory = self._retrieve_index_memory(query, args, kwargs)
        payload = self._build_payload(kwargs, memory)
        res = self.recall(query, *args, payload=payload, **kwargs)

        # if user is requesting to preview the response, then return only the preview result
        if kwargs.get("preview"):
            if self.auto_print:
                UserMessage(str(res), style="text")
            return res

        ### --- asses memory update --- ###

        self._append_interaction_to_memory(query, res)

        # WARN: DO NOT PROCESS THE RES BY REMOVING `<<<` AND `>>>` TAGS

        if self.auto_print:
            UserMessage(str(res), style="text")
        return res

    def _apply_truncation_overrides(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        dynamic_truncation_percentage = kwargs.get(
            "truncation_percentage", self.truncation_percentage
        )
        dynamic_truncation_type = kwargs.get("truncation_type", self.truncation_type)
        return {
            **kwargs,
            "truncation_percentage": dynamic_truncation_percentage,
            "truncation_type": dynamic_truncation_type,
        }

    def _retrieve_index_memory(self, query: Symbol, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if self.index is None:
            return None

        memory_split = self._memory.split(self.marker)
        memory_shards = []
        for shard in memory_split:
            if shard.strip() == "":
                continue
            memory_shards.append(shard)

        length_memory_shards = len(memory_shards)
        if length_memory_shards > 5:
            memory_shards = memory_shards[:2] + memory_shards[-3:]
        elif length_memory_shards > 3:
            retained = memory_shards[-(length_memory_shards - 2) :]
            memory_shards = memory_shards[:2] + retained

        search_query = query | "\n" | "\n".join(memory_shards)
        if kwargs.get("use_seo_opt"):
            search_query = self.seo_opt("[Query]:" | search_query)
        memory = self.index(search_query, *args, **kwargs)

        if "raw_result" in kwargs:
            UserMessage(str(memory), style="text")
        return memory

    def _build_payload(self, kwargs: dict[str, Any], memory) -> str:
        payload = ""
        if "payload" in kwargs:
            payload = f"[Conversation Payload]:\n{kwargs.pop('payload')}\n"

        index_memory = ""
        if memory:
            index_memory = f"[Index Retrieval]:\n{str(memory)[:1500]}\n"
        return f"{index_memory}{payload}"

    def _append_interaction_to_memory(self, query: Symbol, res: Symbol) -> None:
        prompt = self.build_tag(self.user_tag, query)
        self.store(prompt)

        self._value = res.value  # save last response
        val = self.build_tag(self.bot_tag, res)
        self.store(val)


