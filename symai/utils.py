import io
import cv2
import base64
import inspect
import sys
import warnings
import functools
import numpy as np
import multiprocessing as mp
from PIL import Image
from pathos.multiprocessing import ProcessingPool as PPool


def encode_frames_file(file_path):
    ext  = file_path.split('.')[-1]
    if ext.lower() == 'jpg' or ext.lower() == 'jpeg' or ext.lower() == 'png':
        return encode_image(file_path)
    elif ext.lower() == 'gif':
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


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        enc_ = base64.b64encode(image_file.read()).decode('utf-8')
        ext  = image_path.split('.')[-1]
        if ext.lower() == 'jpg':
            ext = 'jpeg'
    return [enc_], ext # return list of base64 encoded images


def parallel(worker=mp.cpu_count()//2):
    def dec(function):
        @functools.wraps(function)
        def _dec(*args, **kwargs):
            with PPool(worker) as pool:
                map_obj = pool.map(function, *args, **kwargs)
            return map_obj
        return _dec
    return dec


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


class Args:
    def __init__(self, skip_none: bool = False, **kwargs):
        # for each key set an attribute
        for key, value in kwargs.items():
            if value is not None or not skip_none:
                if not key.startswith('_'):
                    setattr(self, key, value)


class CustomUserWarning:
    def __init__(self, message: str, stacklevel: int = 1) -> None:
        caller   = inspect.getframeinfo(inspect.stack()[stacklevel][0])
        lineno   = caller.lineno
        filename = caller.filename
        filename = filename[filename.find('symbolicai'):]
        print(f"{filename}:{lineno}: {UserWarning.__name__}: {message}", file=sys.stderr)

