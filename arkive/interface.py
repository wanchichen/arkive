import os
from typing import List, Tuple
from arkive.archive import Arkive
from arkive.audio_read import wav_audio_read, audio_read_local, audio_read_remote
from arkive.definitions import AudioRead
from arkive.utils import check_file_type_func, is_url_regex

def check_file_type(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
) -> Tuple[str, str]:

    with open(archive_path, 'rb') as f:
        f.seek(start_offset)

        header_size = min(file_size, 16)
        header = f.read(header_size)

    return check_file_type_func(header)

def audio_read(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
    start_time: int = None, 
    end_time: int = None
) -> AudioRead:

    if os.path.exists(archive_path):
        return audio_read_local(
            archive_path,
            start_offset,
            file_size,
            start_time,
            end_time
        )

    else:
        return audio_read_remote(
            archive_path,
            start_offset,
            file_size,
            start_time,
            end_time
        )


def create_arkive(
    path: str,
    audio_files: List[str],
    convert_to: str = "flac",
    show_progress: bool = True
) -> Arkive:

    ark = Arkive(path)
    ark.append(
        audio_files,
        convert_to,
        show_progress
    )

    return ark
