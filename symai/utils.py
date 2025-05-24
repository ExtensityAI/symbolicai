from __future__ import annotations

import base64
import inspect
import json
import os
import warnings
from dataclasses import dataclass

import cv2
import httpx
import numpy as np
from box import Box
from PIL import Image
from transformers.models.detr.image_processing_detr_fast import defaultdict

from .misc.console import ConsoleStyle


def encode_media_frames(file_path):
    ext = file_path.split('.')[-1]
    if ext.lower() == 'jpg' or ext.lower() == 'jpeg' or ext.lower() == 'png' or ext.lower() == 'webp':
        if file_path.startswith('http'):
            return encode_image_url(file_path)
        return encode_image_local(file_path)
    elif ext.lower() == 'gif':
        if file_path.startswith('http'):
            raise ValueError("GIF files from URLs are not supported. Please download the file and try again.")

        ext = 'jpeg'
        # get frames from gif
        base64Frames = []
        with Image.open(file_path) as frames:
            for frame in range(frames.n_frames):
                frames.seek(frame)
                # get the image as bytes in memory (using BytesIO)
                current_frame = np.array(frames.convert('RGB'))
                _, buffer = cv2.imencode(".jpg", current_frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        return base64Frames, ext
    elif ext.lower() == 'mp4' or ext.lower() == 'avi' or ext.lower() == 'mov':
        if file_path.startswith('http'):
            raise ValueError("Video files from URLs are not supported. Please download the file and try again.")

        ext = 'jpeg'
        video = cv2.VideoCapture(file_path)
        # get frames from video
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        video.release()
        return base64Frames, ext
    else:
        raise ValueError(f"File extension {ext} not supported")


def encode_image_local(image_path):
    ext = image_path.split('.')[-1]
    # cast `jpg` to `jpeg` to avoid API issues
    if ext.lower() == 'jpg':
        ext = 'jpeg'

    assert ext.lower() in ['jpg', 'jpeg', 'png', 'webp'], f"File extension '{ext}' not supported! Available extensions: ['jpg', 'jpeg', 'png', 'webp']"

    with open(image_path, "rb") as image_file:
        enc_im = base64.b64encode(image_file.read()).decode('utf-8')

    return [enc_im], ext


def encode_image_url(image_url):
    ext = image_url.split('.')[-1]
    # cast `jpg` to `jpeg` to avoid API issues
    if ext.lower() == 'jpg':
        ext = 'jpeg'

    assert ext.lower() in ['jpg', 'jpeg', 'png', 'webp'], f"File extension '{ext}' not supported! Available extensions: ['jpg', 'jpeg', 'png', 'webp']"

    enc_im = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    return [enc_im], ext


def ignore_exception(exception=Exception, default=None):
    """ Decorator for ignoring exception from a function
    e.g.   @ignore_exception(DivideByZero)
    e.g.2. ignore_exception(DivideByZero)(Divide)(2/0)
    """
    def dec(function):
        def _dec(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except exception:
                return default
        return _dec
    return dec


def prep_as_str(x):
    return f"'{str(x)}'" if ignore_exception()(int)(str(x)) is None else str(x)


def deprecated(message):
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator


def toggle_test(enabled: bool = True, default = None):
    def test_decorator(func):
        def test_func(*args, **kwargs):
            if enabled:
                return func(*args, **kwargs)
            else:
                return default
        return test_func
    return test_decorator


class Args:
    def __init__(self, skip_none: bool = False, **kwargs):
        # for each key set an attribute
        for key, value in kwargs.items():
            if value is not None or not skip_none:
                if not key.startswith('_'):
                    setattr(self, key, value)


class CustomUserWarning:
    def __init__(self, message: str, stacklevel: int = 1, raise_with: Exception | None = None) -> None:
        if os.environ.get('SYMAI_WARNINGS', '1') == '1':
            caller   = inspect.getframeinfo(inspect.stack()[stacklevel][0])
            lineno   = caller.lineno
            filename = caller.filename
            filename = filename[filename.find('symbolicai'):]
            with ConsoleStyle('warn') as console:
                console.print(f"{filename}:{lineno}: {UserWarning.__name__}: {message}")
        # Always raise the warning if raise_with is provided
        if raise_with is not None:
            raise raise_with(message)


# Function to format bytes to a human-readable string
def format_bytes(bytes):
    if bytes < 1024:
        return f"{bytes} bytes"
    elif bytes < 1048576:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1073741824:
        return f"{bytes / 1048576:.2f} MB"
    else:
        return f"{bytes / 1073741824:.2f} GB"


@dataclass
class RuntimeInfo:
    total_elapsed_time: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    total_calls: int
    total_tokens: int
    cost_estimate: float

    def __add__(self, other):
        add_elapsed_time = other.total_elapsed_time if hasattr(other, "total_elapsed_time") else 0
        add_prompt_tokens = (
            other.prompt_tokens if hasattr(other, "prompt_tokens") else 0
        )
        add_completion_tokens = (
            other.completion_tokens if hasattr(other, "completion_tokens") else 0
        )
        add_total_tokens = other.total_tokens if hasattr(other, "total_tokens") else 0
        add_cost_estimate = (
            other.cost_estimate if hasattr(other, "cost_estimate") else 0
        )
        add_cached_tokens = (
            other.cached_tokens if hasattr(other, "cached_tokens") else 0
        )
        add_reasoning_tokens = (
            other.reasoning_tokens if hasattr(other, "reasoning_tokens") else 0
        )
        add_total_calls = (
            other.total_calls if hasattr(other, "total_calls") else 0
        )

        return RuntimeInfo(
            total_elapsed_time=self.total_elapsed_time + add_elapsed_time,
            prompt_tokens=self.prompt_tokens + add_prompt_tokens,
            completion_tokens=self.completion_tokens + add_completion_tokens,
            reasoning_tokens=self.reasoning_tokens + add_reasoning_tokens,
            cached_tokens=self.cached_tokens + add_cached_tokens,
            total_calls=self.total_calls + add_total_calls,
            total_tokens=self.total_tokens + add_total_tokens,
            cost_estimate=self.cost_estimate + add_cost_estimate,
        )

    @staticmethod
    def from_tracker(tracker: 'MetadataTracker', total_elapsed_time: float = 0):
        if len(tracker.metadata) > 0:
            try:
                return RuntimeInfo.from_usage_stats(tracker.usage, total_elapsed_time)
            except Exception as e:
                raise e
                CustomUserWarning(f"Failed to parse metadata; returning empty RuntimeInfo: {e}")
                return RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
        return RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)

    @staticmethod
    def from_usage_stats(usage_stats: dict | None, total_elapsed_time: float = 0):
        if usage_stats is not None:
            usage_per_engine = {}
            for (engine_name, model_name), data in usage_stats.items():
                data = Box(data)
                usage_per_engine[(engine_name, model_name)] = RuntimeInfo(
                    total_elapsed_time=total_elapsed_time,
                    prompt_tokens=data.usage.prompt_tokens,
                    completion_tokens=data.usage.completion_tokens,
                    reasoning_tokens=getattr(data.usage, 'reasoning_tokens', 0),
                    cached_tokens=data.prompt_breakdown.cached_tokens,
                    total_calls=data.usage.total_calls,
                    total_tokens=data.usage.total_tokens,
                    cost_estimate=0, # Placeholder for cost estimate
                )
            return usage_per_engine
        return RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)

    @staticmethod
    def estimate_cost(info: RuntimeInfo, f_pricing: callable, **kwargs) -> RuntimeInfo:
        return RuntimeInfo(
            total_elapsed_time=info.total_elapsed_time,
            prompt_tokens=info.prompt_tokens,
            completion_tokens=info.completion_tokens,
            reasoning_tokens=info.reasoning_tokens,
            cached_tokens=info.cached_tokens,
            total_calls=info.total_calls,
            total_tokens=info.total_tokens,
            cost_estimate=f_pricing(info, **kwargs),
        )
