import base64
import inspect
import json
import os
import warnings

import cv2
import httpx
import numpy as np
from PIL import Image

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
    def __init__(self, message: str, stacklevel: int = 1) -> None:
        if os.environ.get('SYMAI_WARNINGS', '1') == '1':
            caller   = inspect.getframeinfo(inspect.stack()[stacklevel][0])
            lineno   = caller.lineno
            filename = caller.filename
            filename = filename[filename.find('symbolicai'):]
            with ConsoleStyle('warn') as console:
                console.print(f"{filename}:{lineno}: {UserWarning.__name__}: {message}")


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
