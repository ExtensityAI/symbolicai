from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from beartype import beartype
from jinja2 import Environment, Template

from ..components import FileReader
from ..few_shots import FewShot
from ..symbol import Expression, Symbol
from ..utils import encode_media_frames


class PromptWeaver(Expression):
    """Context manager for prompt composition to ensure cleanup."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Element._default_context = Context()
        return False

    def compile(self, chain: List[Element]) -> str:
        """Compile the chain."""
        try:
            chain.reverse() # mimic a deque
            element = chain.pop().symbol
            while chain:
                next_element = chain.pop().symbol
                element = element | next_element
            Element._default_context = Context()
            return element
        except Exception as e:
            raise ValueError(f"Error compiling the chain: {e}")
        finally:
            # reset the default context anyway since we might continue to use the prompt weaver and create a new chain
            Element._default_context = Context()

@dataclass
class Context:
    """Track state during prompt composition."""
    system_is_set: bool = False
    elements: List[Element] = field(default_factory=list)

    def remove_system_element(self):
        """Removes the system element from the list."""
        for element in self.elements:
            if isinstance(element, System):
                self.elements.remove(element)
                break
        self.system_is_set = False


class Element(ABC):
    """
    Base class for all elements of the prompt expression language. We're acting on a symbols without a neuro-symbolic engine to compose the prompt by chaining elements together, each element modifying the previous one. Note that order will matter since order dictates the arrangement. By default, we're using the | operator to chain elements together and enclose the specific element within xml tags.
    """
    _default_context = Context()
    def __init__(self):
        self.context = self._default_context
        self.xml_tags = None
        self.env = Environment()
        self._symbol = Symbol(None) # empty symbol

    @abstractmethod
    def validate_element(self) -> bool:
        """Ensure that the element is valid based on a callable or the default implementation."""
        pass

    @abstractmethod
    def validate_composition(self, other: Element) -> bool:
        """
        Ensure that the composition of the element with another element is valid. For instance, having two elements that represent the same thing would be invalid or having two system messages in the same prompt without overwritting the initial one would be invalid.
        """
        pass

    def apply_xml_tags(self, content: str, metadata: Optional[Dict[str, str]]=None) -> str:
        if self.xml_tags is None:
            return content
        left_tag, right_tag = self.xml_tags
        if metadata is None:
            return f"\n{left_tag}\n{content}\n{right_tag}\n"
        return f"\n{left_tag}\n{json.dumps(metadata)}\n{content}\n{right_tag}\n"

    def __or__(self, other: Element) -> Symbol:
        """Compose the element with another element."""
        if isinstance(other, Element):
            if self.validate_composition(other):
                other.context = self.context
                return self
            raise ValueError("Invalid composition of elements.")
        else:
            raise ValueError("Invalid type for composition. An element is required.")

    def __str__(self) -> str:
        """
        Return the string representation of the element.
        """
        return str(self._symbol)


@beartype
class System(Element):
    """
    Represents the system's output.
    """
    def __init__(
        self,
        content: str,
        fn_element_validation: Optional[Callable] = None,
        fn_composition_validation: Optional[Callable] = None,
        *,
        refs: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
        xml_tags: Optional[Tuple[str, str]] = ("<system>", "</system>")
    ):
        super().__init__()
        self.content = Template(content)
        self.fn_element_validation = fn_element_validation
        self.fn_composition_validation = fn_composition_validation
        self.overwrite = overwrite
        self.refs = refs
        self.xml_tags = xml_tags

        self.env.parse(self.content)
        self.content = self.content.render(**self.refs) if self.refs is not None else self.content.render()
        self.content = self.apply_xml_tags(self.content)
        self._symbol = Symbol(self.content)
        # validate the element
        if self.overwrite:
            self.context.remove_system_element()
        if not self.validate_element():
            raise ValueError("Invalid system input.")

        self.context.elements.insert(0, self)

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    def validate_element(self) -> bool:
        if self.fn_element_validation is not None:
            return self.fn_element_validation(self.symbol)
        if not self.context.system_is_set:
            self.context.system_is_set = True
            return True
        return False

    def validate_composition(self, other: Element) -> bool:
        if self.fn_composition_validation is not None:
            return self.fn_composition_validation(self.symbol, other.value)
        return True


@beartype
class User(Element):
    """
    Represents the user's input.
    """
    def __init__(
        self,
        content: str,
        fn_element_validation: Optional[Callable] = None,
        fn_composition_validation: Optional[Callable] = None,
        *,
        refs: Optional[Dict[str, str]] = None,
        xml_tags: Optional[Tuple[str, str]] = ("<user>", "</user>")
    ):
        super().__init__()
        self.content = Template(content)
        self.fn_element_validation = fn_element_validation
        self.fn_composition_validation = fn_composition_validation
        self.refs = refs
        self.xml_tags = xml_tags

        self.env.parse(self.content)
        self.content = self.content.render(**self.refs) if self.refs is not None else self.content.render()
        self.content = self.apply_xml_tags(self.content)
        self._symbol = Symbol(self.content)
        if not self.validate_element():
            raise ValueError("Invalid user input.")

        self.context.elements.append(self)

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    def validate_element(self) -> bool:
        if self.fn_element_validation is not None:
            return self.fn_element_validation(self.symbol)
        return True

    def validate_composition(self, other: Element) -> bool:
        if self.fn_composition_validation is not None:
            return self.fn_composition_validation(self.symbol, other.value)
        return True


@beartype
class Media(Element):
    """Represents a media element (image, video, audio) in the prompt."""
    def __init__(
        self,
        source: str,
        media_type: str = "image",
        fn_element_validation: Optional[Callable] = None,
        fn_composition_validation: Optional[Callable] = None,
        *,
        ref: Optional[str] = None,
        max_frames: int = 10,
        xml_tags: Optional[Tuple[str, str]] = ("<media>", "</media>")
    ):
        super().__init__()
        self.source = source
        self.media_type = media_type
        self.fn_element_validation = fn_element_validation
        self.fn_composition_validation = fn_composition_validation
        self.ref = ref
        self.xml_tags = xml_tags
        self.max_frames = max_frames

        try:
            assert self.media_type in ["image", "video", "gif"], f"Unsupported media type: {self.media_type}. Supported types: ['image', 'video', 'gif']"
            assert Path(self.source).exists(), f"File not found: {self.source}"
            self.frames, self.extension = encode_media_frames(source)
            assert self.extension in ["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "gif"], f"Unsupported file extension: {self.extension}. Supported extensions: ['.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi', '.mov', '.gif']"
            if len(self.frames) > self.max_frames and (self.media_type == "video" or self.media_type == "gif"):
                step = len(self.frames) // self.max_frames
                self.frames = self.frames[::step]
        except Exception as e:
            raise ValueError(f"Failed to encode media from source: {e}")

        self.metadata = json.dumps({
            "ref": self.ref,
            "type": self.media_type,
            "extension": self.extension,
            "source": self.source,
            }
        )
        self.content = self.apply_xml_tags(self.frames, self.metadata)
        self._symbol = Symbol(self.content)

        if not self.validate_element():
            raise ValueError("Invalid media element.")

        self.context.elements.append(self)

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    def validate_element(self) -> bool:
        if self.fn_element_validation is not None:
            return self.fn_element_validation(self.symbol)
        if not len(self.frames) > 0:
            return False
        return True

    def validate_composition(self, other: Element) -> bool:
        if self.fn_composition_validation is not None:
            return self.fn_composition_validation(self.symbol, other.value)
        return True


@beartype
class Document(Element):
    """Represents a document element (text, PDF, or custom content) in the prompt with citation support."""
    def __init__(
        self,
        source: str,
        media_type: str = "document",
        fn_element_validation: Optional[Callable] = None,
        fn_composition_validation: Optional[Callable] = None,
        *,
        ref: Optional[str] = None,
        xml_tags: Optional[Tuple[str, str]] = ("<document>", "</document>"),
        citations_enabled: bool = False,
    ):
        super().__init__()
        self.source = source
        self.media_type = media_type
        self.fn_element_validation = fn_element_validation
        self.fn_composition_validation = fn_composition_validation
        self.ref = ref
        self.xml_tags = xml_tags
        self.citations_enabled = citations_enabled
        self.reader = FileReader()

        try:
            assert self.media_type in ["document"], f"Unsupported media type: {self.media_type}. Supported types: ['document']"
            assert Path(self.source).exists(), f"File not found: {self.source}"
            self.extension = Path(self.source).suffix.strip(".")
            assert self.extension in ["txt", "pdf", "md"], f"Unsupported file extension: {self.extension}. Supported extensions: ['.txt', '.pdf', '.md']"

            self.metadata = {
                "ref": self.ref,
                "type": self.media_type,
                "extension": self.extension,
                "citations": {"enabled": self.citations_enabled},
            }

            self.content = self.reader(self.source).value
            self.content = self.apply_xml_tags(self.content, self.metadata)
            self._symbol = Symbol(self.content)

        except Exception as e:
            raise ValueError(f"Failed to create document element: {e}")

        if not self.validate_element():
            raise ValueError("Invalid document element.")

        self.context.elements.append(self)

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    def validate_element(self) -> bool:
        if self.fn_element_validation is not None:
            return self.fn_element_validation(self.symbol)
        return True

    def validate_composition(self, other: Element) -> bool:
        if self.fn_composition_validation is not None:
            return self.fn_composition_validation(self.symbol, other.value)
        return True


@beartype
class Examples(Element):
    def __init__(
        self,
        examples: Union[List[str], FewShot],
        fn_element_validation: Optional[Callable] = None,
        fn_composition_validation: Optional[Callable] = None,
        *,
        xml_tags: Optional[Tuple[str, str]] = ("<examples>", "</examples>")
    ):
        super().__init__()
        self.examples = examples.value if isinstance(examples, FewShot) else examples
        self.fn_element_validation = fn_element_validation
        self.fn_composition_validation = fn_composition_validation
        self.xml_tags = xml_tags

        self.content = "\n".join(self.examples)
        self.content = self.apply_xml_tags(self.content)
        self._symbol = Symbol(self.content)
        if not self.validate_element():
            raise ValueError("Invalid examples element.")

        self.context.elements.append(self)

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    def validate_element(self) -> bool:
        if self.fn_element_validation is not None:
            return self.fn_element_validation(self.symbol)
        return len(self.examples) > 0

    def validate_composition(self, other: Element) -> bool:
        if self.fn_composition_validation is not None:
            return self.fn_composition_validation(self.symbol, other.value)
        return True

# @TODO
# add tool
# add function
# add interface
# add reasoning (a la r1, o1)
