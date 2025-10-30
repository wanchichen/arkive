from dataclasses import dataclass
import numpy as np

@dataclass
class ArchiveRead:
    file_type: str = None
    modality: str = None

@dataclass
class AudioRead(ArchiveRead):
    sample_rate: int = None
    array: np.ndarray = None

