# Speech Engine (Whisper)

To transcribe audio files, we can perform speech transcription using `whisper`. The following example demonstrates how to transcribe an audio file and return the text:

```python
from symai.interfaces import Interface

speech = Interface('whisper')
res = speech('examples/audio.mp3')
```
