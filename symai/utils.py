from __future__ import annotations

import base64
import importlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from box import Box
from PIL import Image, ImageSequence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from symai.components import MetadataTracker


def encode_media_frames(file_path):
    ext = file_path.split(".")[-1].lower()
    if ext == "jpg" or ext == "jpeg" or ext == "png" or ext == "webp":
        if file_path.startswith("http"):
            return encode_image_url(file_path)
        return encode_image_local(file_path)
    if ext == "gif":
        if file_path.startswith("http"):
            msg = "GIF files from URLs are not supported. Please download the file and try again."
            raise ValueError(msg)

        base64_frames = []
        with Image.open(file_path) as frames:
            for frame in ImageSequence.Iterator(frames):
                buffer = io.BytesIO()
                frame.convert("RGB").save(buffer, format="JPEG")
                base64_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        return base64_frames, "jpeg"
    if ext == "mp4" or ext == "avi" or ext == "mov":
        if file_path.startswith("http"):
            msg = "Video files from URLs are not supported. Please download the file and try again."
            raise ValueError(msg)

        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            msg = (
                "Video frame extraction for mp4/avi/mov requires the optional OpenCV "
                "dependency. Install it with `symbolicai[video]`."
            )
            raise ImportError(msg) from exc

        video = cv2.VideoCapture(file_path)
        base64_frames = []
        try:
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        finally:
            video.release()
        return base64_frames, "jpeg"
    msg = f"File extension {ext} not supported"
    raise ValueError(msg)


def encode_image_local(image_path):
    ext = image_path.split(".")[-1]
    # cast `jpg` to `jpeg` to avoid API issues
    if ext.lower() == "jpg":
        ext = "jpeg"

    assert ext.lower() in ["jpg", "jpeg", "png", "webp"], (
        f"File extension '{ext}' not supported! Available extensions: ['jpg', 'jpeg', 'png', 'webp']"
    )

    with Path(image_path).open("rb") as image_file:
        enc_im = base64.b64encode(image_file.read()).decode("utf-8")

    return [enc_im], ext


def encode_image_url(image_url):
    ext = image_url.split(".")[-1]
    # cast `jpg` to `jpeg` to avoid API issues
    if ext.lower() == "jpg":
        ext = "jpeg"

    assert ext.lower() in ["jpg", "jpeg", "png", "webp"], (
        f"File extension '{ext}' not supported! Available extensions: ['jpg', 'jpeg', 'png', 'webp']"
    )

    enc_im = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    return [enc_im], ext


def semassert(condition: bool, message: str = ""):
    """
    Weak assertion for semantic operations that informs about model limitations.
    Shows a warning instead of failing the test when assertions fail.
    """
    if not condition:
        base_msg = "Assertion failed due to model capability limitations in handling this type of semantic computation"
        full_msg = f"{base_msg}. {message}" if message else base_msg
        logger.warning("⚠️  SEMANTIC ASSERT: %s", full_msg)
        return False
    return True


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
    extras: dict[str, Any] = field(default_factory=dict)

    def __add__(self, other):
        add_elapsed_time = other.total_elapsed_time if hasattr(other, "total_elapsed_time") else 0
        add_prompt_tokens = other.prompt_tokens if hasattr(other, "prompt_tokens") else 0
        add_completion_tokens = (
            other.completion_tokens if hasattr(other, "completion_tokens") else 0
        )
        add_total_tokens = other.total_tokens if hasattr(other, "total_tokens") else 0
        add_cost_estimate = other.cost_estimate if hasattr(other, "cost_estimate") else 0
        add_cached_tokens = other.cached_tokens if hasattr(other, "cached_tokens") else 0
        add_reasoning_tokens = other.reasoning_tokens if hasattr(other, "reasoning_tokens") else 0
        add_total_calls = other.total_calls if hasattr(other, "total_calls") else 0
        extras = other.extras if hasattr(other, "extras") else {}
        merged_extras = {**(self.extras or {})}
        for key, value in (extras or {}).items():
            if (
                key in merged_extras
                and isinstance(merged_extras[key], (int, float))
                and isinstance(value, (int, float))
            ):
                merged_extras[key] += value
            else:
                merged_extras[key] = value

        return RuntimeInfo(
            total_elapsed_time=self.total_elapsed_time + add_elapsed_time,
            prompt_tokens=self.prompt_tokens + add_prompt_tokens,
            completion_tokens=self.completion_tokens + add_completion_tokens,
            reasoning_tokens=self.reasoning_tokens + add_reasoning_tokens,
            cached_tokens=self.cached_tokens + add_cached_tokens,
            total_calls=self.total_calls + add_total_calls,
            total_tokens=self.total_tokens + add_total_tokens,
            cost_estimate=self.cost_estimate + add_cost_estimate,
            extras=merged_extras,
        )

    @staticmethod
    def from_tracker(tracker: MetadataTracker, total_elapsed_time: float = 0):
        if len(tracker.metadata) > 0:
            try:
                return RuntimeInfo.from_usage_stats(tracker.usage, total_elapsed_time)
            except Exception as e:
                msg = f"Failed to parse metadata: {e}"
                raise ValueError(msg) from e
        return RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0, {})

    @staticmethod
    def from_usage_stats(usage_stats: dict | None, total_elapsed_time: float = 0):
        if usage_stats is not None:
            usage_per_engine = {}
            for (engine_name, model_name), data in usage_stats.items():
                #!: This object interacts with `MetadataTracker`; its fields are mandatory and handled there
                data_box = Box(data)
                usage_per_engine[(engine_name, model_name)] = RuntimeInfo(
                    total_elapsed_time=total_elapsed_time,
                    prompt_tokens=data_box.usage.prompt_tokens,
                    completion_tokens=data_box.usage.completion_tokens,
                    reasoning_tokens=data_box.completion_breakdown.reasoning_tokens,
                    cached_tokens=data_box.prompt_breakdown.cached_tokens,
                    total_calls=data_box.usage.total_calls,
                    total_tokens=data_box.usage.total_tokens,
                    cost_estimate=0,  # Placeholder for cost estimate
                    extras=data.get("extras", {}),
                )
            return usage_per_engine
        return RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0, {})

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
            extras=info.extras,
        )
