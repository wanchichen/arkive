from typing import Tuple
import numpy as np
import re

def normalize(audio: np.ndarray, target_peak=1.0) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak > 0:
        normalization_factor = target_peak / peak
        audio_normalized = audio * normalization_factor
    else:
        audio_normalized = audio
    return audio_normalized

def _float_to_int16(audio_float: np.ndarray) -> np.ndarray:
    """Convert float32 audio to int16 PCM."""
    # Clip to [-1.0, 1.0] range
    audio_float = np.clip(audio_float, -1.0, 1.0)
    
    # Convert to 16-bit integer
    audio_int16 = (audio_float * 32767).astype(np.int16)
    
    return audio_int16
    
def _float_to_int32(audio_float: np.ndarray) -> np.ndarray:
    """Convert float32 audio to int32 PCM."""
    # Clip to [-1.0, 1.0] range
    audio_float = np.clip(audio_float, -1.0, 1.0)
    
    # Convert to 32-bit integer
    audio_int32 = (audio_float * 2147483647).astype(np.int32)
    
    return audio_int32

def _float_to_float64(audio_float: np.ndarray) -> np.ndarray:
    """Convert float32 audio to float64."""
    return audio_float.astype(np.float64)

def check_file_type_func(
    header: bytes
) -> Tuple[str, str]:

    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
        return ('wav', 'audio')

    else:
        return ('flac', 'audio')

def is_url_regex(text):
    # This regex is a common starting point for URL validation, but may not cover all edge cases.
    url_pattern = re.compile(
        r'^(https?://)?'  # Optional http or https
        r'([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}'  # Domain name
        r'(:\d+)?'  # Optional port
        r'(/([a-zA-Z0-9-._~:/?#@!$&\'()*+,;=]*))?$'  # Optional path, query, fragment
    )
    return bool(url_pattern.match(text))