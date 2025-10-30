#!/usr/bin/env python3

import io
import numpy as np
import requests
import soundfile as sf

from arkive.definitions import AudioRead
from arkive.utils import check_file_type_func

def wav_audio_read(archive_path: str, start_offset: int, file_size: int, start_time: int, end_time: int) -> AudioRead:
    with open(archive_path, 'rb') as f:
        f.seek(start_offset)

        # Read the WAV header (first 44 bytes for standard WAV)
        # We'll read a bit more to handle potential extended headers
        header = f.read(100)

        # Use wave module to parse the header
        header_io = io.BytesIO(header_data)
        with wave.open(header_io, 'rb') as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()

            # Get the data chunk position
            # The wave module has positioned us right at the start of audio data
            header_io.seek(0)
            temp_wav = wave.open(header_io, 'rb')
            temp_wav.readframes(0)  # This positions us at the start of data
            data_start_offset = header_io.tell()

        # Calculate frame positions
        if start_time is None:
            start_frame = 0
        else:
            start_frame = int(start_time * framerate)

        if end_frame is None:
            end_frame = n_frames
        else:
            end_frame = int(end_time * framerate)
        
        # Ensure we don't go beyond the file
        start_frame = max(0, min(start_frame, n_frames))
        end_frame = max(start_frame, min(end_frame, n_frames))
        
        num_frames_to_read = end_frame - start_frame
        # Calculate byte position in the archive
        frame_size = n_channels * sampwidth
        byte_offset_in_wav = data_start_offset + (start_frame * frame_size)
        absolute_offset = file_offset + byte_offset_in_wav
        
        # Seek to the exact position and read only what we need
        f.seek(absolute_offset)
        num_bytes = num_frames_to_read * frame_size
        audio_bytes = f.read(num_bytes)

        return generic_audio_read(
            audio_bytes,
            "wav"
        )

def generic_audio_read(audio_bytes: bytes, file_type: str, start_time: int, end_time: int) -> AudioRead:
    audio_file_like = io.BytesIO(audio_bytes)

    audio, sr = sf.read(audio_file_like)

    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=1)

    if start_time is None:
        start_frame = 0
    else:
        start_frame = start_time * sr

    if end_time is None:
        end_frame = audio.shape[0]
    else:
        end_frame = end_time * sr

    return AudioRead(
        file_type=file_type,
        modality="audio",
        sample_rate=sr,
        array=audio[start_frame:end_frame, :]
    )

    
def audio_read_remote(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
    start_time: int = None, 
    end_time: int = None
) -> AudioRead:
    http_headers = {'Range': f'bytes={start_offset}-{start_offset+file_size+1}'}
    response = requests.get(archive_path, headers=http_headers, stream=True)

    # Check if server supports range requests
    if response.status_code not in [200, 206]:
        print("Warning: Server doesn't support range requests.")
        raise Exception(f"Server returned status code {response.status_code}")

    file_type, modality = check_file_type_func(response.content)
    assert modality == "audio", f"Expected audio modality, got {modality} from {archive_path} at byte {start_offset}"

    if file_type != "wav":
        return generic_audio_read(
            response.content,
            file_type,
            start_time,
            end_time
        )
    else:
        raise Exception("Not implemented for wavs yet")

def audio_read_local(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
    start_time: int = None, 
    end_time: int = None
) -> AudioRead:

    with open(archive_path, 'rb') as f:
        f.seek(start_offset)

        header_size = min(file_size, 16)
        header = f.read(header_size)

        file_type, modality = check_file_type_func(header)

        assert modality == "audio", f"Expected audio modality, got {modality} from {archive_path} at byte {start_offset}"

        if file_type != "wav":
            f.seek(start_offset)
            audio_bytes = f.read(file_size)

            return generic_audio_read(
                audio_bytes,
                file_type,
                start_time,
                end_time
            )

    if file_type == "wav":
        return wav_audio_read(
            archive_path,
            start_offset,
            file_size,
            start_time,
            end_time
        )