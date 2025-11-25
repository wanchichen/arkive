#!/usr/bin/env python3
"""
Audio Archive Tool
Converts mixed audio formats to 16-bit PCM FLAC and stores in a single archive with Parquet metadata.
"""

import io
import os
import itertools
import warnings
from pathlib import Path
from typing import (
    List, Optional, Tuple, Union, Iterator,
    Dict, Any, overload, BinaryIO
)
import numpy as np
import pandas as pd
import soundfile as sf
import subprocess
import tempfile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from arkive.audio_read import generic_audio_read
from arkive.definitions import AudioRead
from arkive.utils import (
    _get_bit_depth_from_subtype,
)


def _convert_audio_data(
    audio_data: np.ndarray,
    sample_rate: int,
    channels: int,
    target_format: str,
    target_bit_depth: int
) -> Tuple[bytes, int]:
    """
    Convert audio data to target format and bit depth.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        channels: Number of channels
        target_format: Target format ('flac', 'wav', etc.)
        target_bit_depth: Target bit depth (16, 32, or 64)
        
    Returns:
        Tuple of (file_data_bytes, samples_count)
    """
    # Reshape if mono
    if channels == 1:
        audio_data = audio_data.reshape(-1, 1)
    
    # Determine subtype
    if target_bit_depth == 16:
        subtype = 'PCM_16'
    elif target_bit_depth == 32:
        subtype = 'PCM_32'
    elif target_bit_depth == 64:
        subtype = 'DOUBLE'
    else:
        raise ValueError(f"Unsupported target_bit_depth: {target_bit_depth}")
    
    # Convert to target format
    out_buf = io.BytesIO()
    out_buf.name = f'temp.{target_format}'
    sf.write(out_buf, audio_data, sample_rate, 
            subtype=subtype, format=target_format.upper())
    out_buf.seek(0)
    file_data = out_buf.read()
    
    return file_data, len(audio_data)


# Static function for multiprocessing (must be at module level for pickle)
def _process_single_audio_file_static(
    audio_file: str, target_format: Optional[str], target_bit_depth: int
) -> Optional[Tuple]:
    """
    Process a single audio file to convert it to the target format and bit depth.
    This function must be at module level to be picklable.
    
    Returns:
        Tuple of (audio_file, file_data, sample_rate, channels, samples, format, bit_depth) or None if failed
    """
    try:
        # Read the file - handle ark format
        if ':' in audio_file and 'ark' in audio_file.lower():
            try:
                import kaldiio
            except ImportError:
                raise ImportError("kaldiio is not installed. Please `pip install kaldiio`")
            sample_rate, audio_data = kaldiio.load_mat(audio_file)
            channels = 1
            orig_bit_depth = 16
            orig_format = 'wav' # write into wav instead of flac, since flac cost more time
            
            buffer = io.BytesIO()
            buffer.name = 'temp.wav'
            sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            orig_file_data = buffer.read()
        else:
            with open(audio_file, 'rb') as f:
                orig_file_data = f.read()
            
            try:
                audio_data, sample_rate = sf.read(audio_file, dtype='float32')
                info = sf.info(audio_file)
                orig_bit_depth = _get_bit_depth_from_subtype(info.subtype)
            except:
                # Fallback - just return None, will be handled by error
                print(f"Warning: Could not read {audio_file} with soundfile, skipping...")
                return None
            
            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[1]
            
            orig_format = Path(audio_file).suffix.lower().lstrip('.')
        
        orig_samples = len(audio_data)
        
        # Determine if conversion is needed
        needs_conversion = False
        if target_format is not None:
            if orig_format != target_format or orig_bit_depth != target_bit_depth:
                needs_conversion = True
        
        if needs_conversion:
            # Convert using helper function
            file_data, samples = _convert_audio_data(
                audio_data, sample_rate, channels, target_format, target_bit_depth
            )
            file_format = target_format
            bit_depth = target_bit_depth
        else:
            file_data = orig_file_data
            file_format = orig_format
            bit_depth = orig_bit_depth
            samples = orig_samples
        
        return (audio_file, file_data, sample_rate, channels, samples, file_format, bit_depth)
    
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


def _process_audio_from_bytes_static(
    audio_item: dict, target_format: Optional[str], target_bit_depth: int
) -> Optional[Tuple]:
    """
    Process audio from bytes (no file I/O needed for input).
    This function must be at module level to be picklable.
    
    Args:
        audio_item: Dict with keys:
            - 'bytes': audio file bytes
            - 'key': unique identifier for this audio
            - 'format': original format (e.g., 'mp3', 'flac', 'wav')
        target_format: Target format for conversion
        target_bit_depth: Target bit depth
        
    Returns:
        Tuple of (key, file_data, sample_rate, channels, samples, format, bit_depth) or None if failed
    """
    try:
        audio_bytes = audio_item['bytes']
        audio_key = audio_item['key']
        orig_format = audio_item['format']
        
        # Create BytesIO buffer
        orig_buffer = io.BytesIO(audio_bytes)
        orig_buffer.name = f'temp.{orig_format}'  # soundfile needs this to infer format
        
        # Read audio from memory
        try:
            audio_data, sample_rate = sf.read(orig_buffer, dtype='float32')
            orig_buffer.seek(0)
            info = sf.info(orig_buffer)
            orig_bit_depth = _get_bit_depth_from_subtype(info.subtype)
        except Exception as e:
            print(f"Warning: Could not read audio {audio_key}: {e}")
            return None
        
        if audio_data.ndim == 1:
            channels = 1
        else:
            channels = audio_data.shape[1]
        
        orig_samples = len(audio_data)
        
        # Determine if conversion is needed
        needs_conversion = False
        if target_format is not None:
            if orig_format != target_format or orig_bit_depth != target_bit_depth:
                needs_conversion = True
        
        if needs_conversion:
            # Convert using helper function
            file_data, samples = _convert_audio_data(
                audio_data, sample_rate, channels, target_format, target_bit_depth
            )
            file_format = target_format
            bit_depth = target_bit_depth
        else:
            # No conversion needed, use original bytes
            file_data = audio_bytes
            file_format = orig_format
            bit_depth = orig_bit_depth
            samples = orig_samples
        
        return (audio_key, file_data, sample_rate, channels, samples, file_format, bit_depth)
    
    except Exception as e:
        print(f"Error processing audio {audio_item.get('key', 'unknown')}: {e}")
        return None


def _process_item_unified(
    item: Union[str, dict], target_format: Optional[str], target_bit_depth: int
) -> Optional[Tuple]:
    """
    Unified processing function that automatically selects the appropriate handler
    based on item type.
    
    Args:
        item: Either a file path (str) or a bytes dict with keys:
            - 'bytes': audio file bytes
            - 'key': unique identifier
            - 'format': original format
        target_format: Target format for conversion
        target_bit_depth: Target bit depth
        
    Returns:
        Tuple of (identifier, file_data, sample_rate, channels, samples, format, bit_depth) or None if failed
    """
    if isinstance(item, str):
        # File path type
        return _process_single_audio_file_static(item, target_format, target_bit_depth)
    elif isinstance(item, dict):
        # Bytes stream type
        return _process_audio_from_bytes_static(item, target_format, target_bit_depth)
    else:
        raise TypeError(f"Unsupported item type: {type(item)}. Expected str (file path) or dict (bytes stream)")


class Arkive:
    """Create and manage audio archives with mixed format support."""
    
    MAGIC_NUMBER = b'AUDARCH1'
    MAX_BIN_SIZE = 32 * 1024 * 1024 * 1024  # default 32 GB in bytes
    
    def __init__(self, archive_dir: str, dataset_name: Optional[str] = None):
        """
        Initialize audio archive.

        Args:
            archive_dir: Path to the archive directory
                archive_dir/
                    audio_arkive_0.bin
                    audio_arkive_1.bin (if over MAX_BIN_SIZE)
                    ...
                    metadata.parquet
        """
        self.archive_path = Path(archive_dir) / "arkive"
        self.metadata_file = Path(archive_dir) / "metadata.parquet"

        self.data = None
        if self.metadata_file.exists():
            self.data = self.get_metadata()
        
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            archive_dir = str(archive_dir)
            archive_dir_parts = archive_dir.split("/")
            if "split_0" in archive_dir_parts:
                self.dataset_name = archive_dir_parts[-3]
            elif archive_dir_parts[-1] == "arkive":
                self.dataset_name = archive_dir_parts[-2]
            else:
                self.dataset_name = None

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)
        
    def _get_bin_file_path(self, bin_index: int) -> Path:
        """
        Get the path for a specific bin file.
        
        Args:
            bin_index: Index of the bin file (0, 1, 2, ...)
            
        Returns:
            Path to the bin file
        """
        return self.archive_path.with_name(f"{self.archive_path.stem}_{bin_index}.bin")
    
    def _get_current_bin_info(self) -> tuple:
        """
        Get the current bin file index and its size.
        
        Returns:
            Tuple of (bin_index, current_size)
        """
        bin_index = 0
        while True:
            bin_path = self._get_bin_file_path(bin_index)
            if not bin_path.exists():
                # If this bin doesn't exist, use the previous one (or 0 if none exist)
                if bin_index == 0:
                    return 0, 0
                else:
                    prev_bin_path = self._get_bin_file_path(bin_index - 1)
                    return bin_index - 1, prev_bin_path.stat().st_size
            bin_index += 1
    
    def _write_result_to_bin(
        self,
        result: Tuple,
        current_bin_file: BinaryIO,
        current_bin_index: int,
        current_offset: int,
        item_type: str,
        show_progress: bool,
        utt_id: str
    ) -> Tuple[BinaryIO, int, int, dict]:
        """
        Write a processed result to bin file and create metadata record.
        
        Args:
            result: Tuple from processing function (identifier, file_data, sample_rate, channels, samples, format, bit_depth)
            current_bin_file: Current bin file handle
            current_bin_index: Current bin file index
            current_offset: Current byte offset in bin file
            item_type: 'file' or 'bytes'
            show_progress: Whether to show progress messages
            
        Returns:
            Tuple of (updated_bin_file, updated_bin_index, updated_offset, metadata_record)
        """
        identifier, file_data, sample_rate, channels, samples, file_format, bit_depth = result
        file_size = len(file_data)
        
        # Check if we need to create a new bin file
        if current_offset + file_size > self.MAX_BIN_SIZE:
            current_bin_file.close()
            current_bin_index += 1
            current_offset = 0
            current_bin_path = self._get_bin_file_path(current_bin_index)
            current_bin_file = open(current_bin_path, 'wb')
            if show_progress:
                print(f"\nCreating new bin file: {current_bin_path.name} (previous bin full)")
        
        # Determine identifier field based on item type
        if item_type == 'file' or (isinstance(identifier, str) and Path(identifier).exists()):
            original_file_path = str(Path(identifier).absolute())
        else:
            original_file_path = identifier  # Use key for bytes type
        
        # Write to bin file
        current_bin_file.write(file_data)
        
        # Create metadata record
        current_bin_path = self._get_bin_file_path(current_bin_index)
        metadata_record = {
            'utt_id': utt_id,
            'original_file_path': original_file_path,
            'bin_index': current_bin_index,
            'path': str(current_bin_path),
            'start_byte_offset': current_offset,
            'file_size_bytes': file_size,
            'sample_rate': sample_rate,
            'channels': channels,
            'length': samples,
            'format': file_format,
            'bit_depth': bit_depth
        }
        
        # Update offset
        current_offset += file_size
        
        return current_bin_file, current_bin_index, current_offset, metadata_record

    def _append_items(
        self,
        items_iterator: Iterator[Union[str, Dict[str, Any]]],
        target_format: Optional[str] = 'flac',
        target_bit_depth: int = 16,
        show_progress: bool = False,
        flush_interval: int = 10,
        num_workers: int = None,
        batch_size: Optional[int] = None,
        item_type: str = 'auto'
    ):
        """
        Unified internal append method that handles all common logic.
        
        Args:
            items_iterator: Iterator yielding unified item format (str for file paths, dict for bytes)
            target_format: Target format for conversion
            target_bit_depth: Target bit depth
            show_progress: Whether to show progress messages
            flush_interval: Flush bin file buffer to disk every N files (0 to disable)
            num_workers: Number of worker processes (default: cpu_count() - 1)
            batch_size: Batch size for processing (None means no batching, mainly for bytes type)
            item_type: 'file' for file paths, 'bytes' for bytes streams, 'auto' for auto-detection
        """
        metadata_records = []
        processed_count = 0
        current_bin_index, current_offset = self._get_current_bin_info()
        
        # Validate parameters
        if target_format is not None:
            target_format = target_format.lower()
            valid_formats = ['flac', 'wav', 'mp3', 'opus']
            if target_format not in valid_formats:
                raise ValueError(f"target_format must be one of {valid_formats} or None, got '{target_format}'")
        
        if target_bit_depth not in [16, 32, 64]:
            raise ValueError(f"target_bit_depth must be 16, 32, or 64, got {target_bit_depth}")
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        # Open the current bin file for appending
        current_bin_path = self._get_bin_file_path(current_bin_index)
        current_bin_file = open(current_bin_path, 'ab')
        
        try:
            # Determine if we need batch processing
            use_batch_processing = (batch_size is not None and batch_size > 0)
            
            if use_batch_processing:
                # Batch processing mode (mainly for bytes type)
                batch = []
                batch_count = 0
                items_with_progress = tqdm(items_iterator, desc="Processing audios") if show_progress else items_iterator
                
                for i, item in enumerate(items_with_progress):
                    batch.append(item)
                    
                    # When batch is full, process it
                    if len(batch) >= batch_size:
                        current_bin_file, current_bin_index, current_offset, processed_count, metadata_records = \
                            self._process_batch(
                                batch,
                                current_bin_file,
                                current_bin_index,
                                current_offset,
                                processed_count,
                                metadata_records,
                                target_format,
                                target_bit_depth,
                                num_workers,
                                show_progress,
                                flush_interval,
                                item_type,
                                batch_count,
                                batch_size
                            )
                        batch_count += 1
                        batch = []  # Clear batch to free memory
                
                # Process remaining batch
                if batch:
                    current_bin_file, current_bin_index, current_offset, processed_count, metadata_records = \
                        self._process_batch(
                            batch,
                            current_bin_file,
                            current_bin_index,
                            current_offset,
                            processed_count,
                            metadata_records,
                            target_format,
                            target_bit_depth,
                            num_workers,
                            show_progress,
                            flush_interval,
                            item_type,
                            batch_count,
                            batch_size
                        )
            else:
                # Non-batch processing mode (for file paths or when batch_size is None)
                # Convert to list if needed for progress bar total
                if isinstance(items_iterator, list):
                    total = len(items_iterator)
                    items_iter = iter(items_iterator)
                elif hasattr(items_iterator, '__len__'):
                    total = len(items_iterator)
                    items_iter = items_iterator
                else:
                    # For iterators without length, we can't show total
                    items_iter = items_iterator
                    total = None
                
                # Use multiprocessing if num_workers > 1
                if num_workers > 1:
                    pool = Pool(processes=num_workers)
                    process_func = partial(
                        _process_item_unified,
                        target_format=target_format,
                        target_bit_depth=target_bit_depth
                    )
                    
                    # Process items in parallel
                    results_iter = pool.imap_unordered(process_func, items_iter)
                    if show_progress:
                        results_iter = tqdm(results_iter, total=total, desc="Processing audio files")
                    
                    for i, result in enumerate(results_iter):
                        if result is None:
                            continue
                        
                        # Write result to bin file and get metadata
                        utt_id = f"{self.dataset_name}_bin{current_bin_index}_{i}"
                        current_bin_file, current_bin_index, current_offset, metadata_record = \
                            self._write_result_to_bin(
                                result, current_bin_file, current_bin_index, current_offset,
                                item_type, show_progress, utt_id
                            )
                        metadata_records.append(metadata_record)
                        processed_count += 1
                        
                        # Periodic flush
                        if flush_interval > 0 and (i + 1) % flush_interval == 0:
                            self._flush_data(current_bin_file, metadata_records, show_progress, i + 1)
                            metadata_records = []
                    
                    pool.close()
                    pool.join()
                else:
                    # Single process mode
                    items_with_progress = tqdm(enumerate(items_iter), total=total, desc="Processing audio files") if show_progress else enumerate(items_iter)
                    for i, item in items_with_progress:
                        try:
                            result = _process_item_unified(item, target_format, target_bit_depth)
                            if result is None:
                                continue
                            
                            # Write result to bin file and get metadata
                            utt_id = f"{self.dataset_name}_bin{current_bin_index}_{i}"
                            current_bin_file, current_bin_index, current_offset, metadata_record = \
                                self._write_result_to_bin(
                                    result, current_bin_file, current_bin_index, current_offset,
                                    item_type, show_progress, utt_id
                                )
                            metadata_records.append(metadata_record)
                            processed_count += 1
                            
                            # Periodic flush
                            if flush_interval > 0 and (i + 1) % flush_interval == 0:
                                self._flush_data(current_bin_file, metadata_records, show_progress, i + 1)
                                metadata_records = []
                        
                        except Exception as e:
                            item_str = item if isinstance(item, str) else item.get('key', 'unknown')
                            print(f"Error processing {item_str}: {e}")
                            continue
        
        finally:
            # Final flush before closing to ensure all data is written
            if flush_interval > 0 and metadata_records:
                self._flush_data(current_bin_file, metadata_records, show_progress, processed_count)
                metadata_records = []
            current_bin_file.close()
        
        # Save remaining metadata if not using flush
        if not flush_interval > 0:
            new_df = pd.DataFrame(metadata_records)
            
            if self.metadata_file.exists():
                # Load existing metadata and append
                existing_df = self.data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                self.data = combined_df
                combined_df.to_parquet(self.metadata_file, index=False)
                
                if show_progress:
                    print(f"\nAppended {len(metadata_records)} files to existing archive!")
            else:
                # Create new metadata file
                new_df.to_parquet(self.metadata_file, index=False)
                self.data = new_df
        
        if show_progress:
            print(f"\nArchive created successfully!")
            print(f"Metadata file: {self.metadata_file}")
            print(f"Total files: {processed_count}")
            
            # Show bin file statistics
            if self.data is not None:
                bin_indices = self.data['bin_index'].unique()
                for bin_idx in sorted(bin_indices):
                    bin_path = self._get_bin_file_path(bin_idx)
                    bin_size = bin_path.stat().st_size if bin_path.exists() else 0
                    file_count = len(self.data[self.data['bin_index'] == bin_idx])
                    print(f"  {bin_path.name}: {bin_size / (1024**3):.2f} GB ({file_count} files)")
                
                total_size = sum((self._get_bin_file_path(idx).stat().st_size 
                                for idx in bin_indices 
                                if self._get_bin_file_path(idx).exists()))
                print(f"Total size: {total_size / (1024**3):.2f} GB")
    
    def _process_batch(
        self,
        batch: List[Union[str, Dict[str, Any]]],
        current_bin_file,
        current_bin_index: int,
        current_offset: int,
        processed_count: int,
        metadata_records: List[dict],
        target_format: Optional[str],
        target_bit_depth: int,
        num_workers: int,
        show_progress: bool,
        flush_interval: int,
        item_type: str,
        batch_count: int,
        batch_size: int
    ):
        """Process a batch of items with parallel processing."""
        
        # Create process pool for this batch
        pool = Pool(processes=num_workers)
        process_func = partial(
            _process_item_unified,
            target_format=target_format,
            target_bit_depth=target_bit_depth
        )
        
        # Process batch in parallel (use imap_unordered to avoid bubble)
        results_iter = pool.imap_unordered(process_func, batch)
        
        for i, result in enumerate(results_iter):
            if result is None:
                continue
            
            # Write result to bin file and get metadata
            global_index = batch_count * batch_size + i
            utt_id = f"{self.dataset_name}_bin{current_bin_index}_{global_index}"
            current_bin_file, current_bin_index, current_offset, metadata_record = \
                self._write_result_to_bin(
                    result, current_bin_file, current_bin_index, current_offset,
                    item_type, show_progress, utt_id
                )
            metadata_records.append(metadata_record)
            processed_count += 1
            
            # Periodic flush
            if flush_interval > 0 and processed_count % flush_interval == 0:
                self._flush_data(current_bin_file, metadata_records, show_progress, processed_count)
                metadata_records = []
        
        pool.close()
        pool.join()
        
        return current_bin_file, current_bin_index, current_offset, processed_count, metadata_records

    @overload
    def append(
        self,
        items: List[str],
        target_format: Optional[str] = 'flac',
        target_bit_depth: int = 16,
        show_progress: bool = False,
        flush_interval: int = 10,
        num_workers: int = None,
        batch_size: Optional[int] = None
    ) -> None: ...
    
    @overload
    def append(
        self,
        items: Iterator[str],
        target_format: Optional[str] = 'flac',
        target_bit_depth: int = 16,
        show_progress: bool = False,
        flush_interval: int = 10,
        num_workers: int = None,
        batch_size: Optional[int] = None
    ) -> None: ...
    
    @overload
    def append(
        self,
        items: Iterator[Dict[str, Any]],
        target_format: Optional[str] = 'flac',
        target_bit_depth: int = 16,
        show_progress: bool = False,
        flush_interval: int = 100,
        num_workers: int = None,
        batch_size: int = 200
    ) -> None: ...
    
    def append(
        self,
        items: Union[List[str], Iterator[str], Iterator[Dict[str, Any]]],
        target_format: Optional[str] = 'flac',
        target_bit_depth: int = 16,
        show_progress: bool = False,
        flush_interval: int = 100,
        num_workers: int = None,
        batch_size: Optional[int] = None
    ):
        """
        Unified append interface that automatically detects input type.
        
        Supports multiple input formats:
        - List[str]: List of file paths
        - Iterator[str]: Iterator of file paths
        - Iterator[dict]: Iterator of bytes dicts with keys:
            - 'bytes': audio file bytes
            - 'key': unique identifier
            - 'format': original format (e.g., 'mp3', 'flac')
            - 'metadata': optional metadata dict
        
        Args:
            items: Input items (file paths or bytes dicts)
            target_format: Target format for conversion ('flac', 'wav', 'mp3', 'opus', or None to retain original)
            target_bit_depth: Target bit depth for conversion (16, 32, or 64). Only applies when target_format is not None.
            show_progress: Whether to show progress messages
            flush_interval: Flush bin file buffer to disk every N files (default: 10 for files, 100 for bytes). Set to 0 to disable periodic flushing.
            num_workers: Number of worker processes for parallel processing (default: cpu_count() - 1, set to 1 to disable)
            batch_size: Batch size for processing (only effective for bytes type, default: 200). Ignored for file paths.
        
        Examples:
            # File path list
            archive.append(['file1.wav', 'file2.mp3'])
            
            # File path iterator
            archive.append(Path('audio_dir').glob('*.wav'))
            
            # Bytes stream iterator
            archive.append(audio_bytes_iterator, batch_size=200)
        """
        # Auto-detect input type
        items_iter = iter(items)
        first_item = next(items_iter, None)
        
        if first_item is None:
            return  # Empty input
        
        # Rebuild iterator (including first item)
        items_iter = itertools.chain([first_item], items_iter)
        
        # Detect type
        if isinstance(first_item, str):
            # File path type
            item_type = 'file'
            if batch_size is not None:
                warnings.warn("batch_size is ignored for file path inputs", UserWarning)
            batch_size = None
        elif isinstance(first_item, dict):
            # Bytes stream type
            item_type = 'bytes'
            if batch_size is None:
                batch_size = 200  # Default batch size for bytes
        else:
            raise TypeError(
                f"Unsupported item type: {type(first_item)}. "
                "Expected str (file path) or dict (bytes stream)"
            )
        
        return self._append_items(
            items_iter,
            target_format=target_format,
            target_bit_depth=target_bit_depth,
            show_progress=show_progress,
            flush_interval=flush_interval,
            num_workers=num_workers,
            batch_size=batch_size,
            item_type=item_type
        )
    
    def append_from_bytes(
        self, 
        audio_bytes_iterator,
        batch_size: int = 200,
        target_format: Optional[str] = 'flac', 
        target_bit_depth: int = 16,
        show_progress: bool = False,
        flush_interval: int = 100,
        num_workers: int = None
    ):
        """
        Append audio from bytes (streaming, memory-efficient).
        
        .. deprecated:: 2.0
            Use :meth:`append` instead. This method will be removed in a future version.
        
        This method processes audio from an iterator of byte streams, which is more
        memory-efficient than loading all audio into memory at once.
        
        Args:
            audio_bytes_iterator: Iterator yielding dicts with keys:
                - 'bytes': audio file bytes
                - 'key': unique identifier
                - 'format': original format (e.g., 'mp3', 'flac')
                - 'metadata': optional metadata dict
            batch_size: Number of audio items to process in each batch (controls memory)
            target_format: Target format for conversion ('flac', 'wav', etc.)
            target_bit_depth: Target bit depth (16, 32, or 64)
            show_progress: Whether to show progress messages
            flush_interval: Flush metadata every N files
            num_workers: Number of worker processes (default: cpu_count() - 1)
        """
        warnings.warn(
            "append_from_bytes is deprecated. Use append() instead. "
            "append_from_bytes will be removed in v2.0",
            DeprecationWarning,
            stacklevel=2
        )
        return self.append(
            audio_bytes_iterator,
            batch_size=batch_size,
            target_format=target_format,
            target_bit_depth=target_bit_depth,
            show_progress=show_progress,
            flush_interval=flush_interval,
            num_workers=num_workers
        )
    
    def _flush_data(
        self,
        current_bin_file: BinaryIO,
        metadata_records: List[dict],
        show_progress: bool,
        file_count: int
    ):
        """
        Flush bin file and metadata to disk.
        
        Args:
            current_bin_file: File handle for current bin file
            metadata_records: List of metadata records to flush
            show_progress: Whether to show progress messages
            file_count: Number of files processed so far
        """
        try:
            current_bin_file.flush()
            os.fsync(current_bin_file.fileno())
            
            if metadata_records:
                new_df = pd.DataFrame(metadata_records)
                if self.metadata_file.exists():
                    existing_df = self.data
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    self.data = combined_df
                    combined_df.to_parquet(self.metadata_file, index=False)
                else:
                    new_df.to_parquet(self.metadata_file, index=False)
                    self.data = new_df
        
        except (OSError, IOError) as e:
            print(f"Warning: Failed to flush data to disk: {e}")
        
    def _read_with_ffmpeg(self, audio_file: str) -> tuple:
        """
        Read audio file using ffmpeg for formats not supported by soundfile.
        
        Args:
            audio_file: Path to input audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Get audio info first
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=sample_rate,channels',
            '-of', 'default=noprint_wrappers=1',
            audio_file
        ]
        
        try:
            probe_output = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT).decode()
            sample_rate = int([line.split('=')[1] for line in probe_output.split('\n') if 'sample_rate' in line][0])
        except:
            sample_rate = 44100  # Default fallback
        
        # Convert to WAV PCM using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            cmd = [
                'ffmpeg', '-i', audio_file,
                '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate),
                '-y', tmp_path,
                '-loglevel', 'error'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Read the converted WAV file
            audio_data, actual_sr = sf.read(tmp_path, dtype='float32')
            sample_rate = actual_sr
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return audio_data, sample_rate
    
    def _write_with_ffmpeg(self, audio_data: np.ndarray, sample_rate: int, 
                          output_path: str, target_format: str, target_bit_depth: int):
        """
        Write audio data using ffmpeg for MP3 and OPUS formats.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz
            output_path: Output file path
            target_format: Target format ('mp3' or 'opus')
            target_bit_depth: Target bit depth (16, 32, or 64)
        """
        # First write to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        
        try:
            # Write as WAV with appropriate bit depth
            if target_bit_depth == 16:
                sf.write(tmp_wav_path, audio_data, sample_rate, subtype='PCM_16', format='WAV')
            elif target_bit_depth == 32:
                sf.write(tmp_wav_path, audio_data, sample_rate, subtype='PCM_32', format='WAV')
            else:  # 64
                sf.write(tmp_wav_path, audio_data, sample_rate, subtype='DOUBLE', format='WAV')
            
            # Convert to target format using ffmpeg
            if target_format == 'mp3':
                # Use high quality MP3 encoding
                bitrate = '320k' if target_bit_depth >= 32 else '192k'
                cmd = [
                    'ffmpeg', '-i', tmp_wav_path,
                    '-codec:a', 'libmp3lame',
                    '-b:a', bitrate,
                    '-y', output_path,
                    '-loglevel', 'error'
                ]
            elif target_format == 'opus':
                # Use high quality OPUS encoding
                bitrate = '256k' if target_bit_depth >= 32 else '128k'
                cmd = [
                    'ffmpeg', '-i', tmp_wav_path,
                    '-codec:a', 'libopus',
                    '-b:a', bitrate,
                    '-y', output_path,
                    '-loglevel', 'error'
                ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
        finally:
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
    
    def get_metadata(self) -> pd.DataFrame:
        """
        Get the metadata DataFrame.
        
        Returns:
            Pandas DataFrame with archive metadata
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        return pd.read_parquet(self.metadata_file)
    
    def extract_file(self, index: int, start_time: int = None, end_time: int = None) -> AudioRead:
        """
        Extract a file from the archive by index.
        
        Args:
            index: Index of the file in the metadata
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
            
        Returns:
            Binary data of the file
        """
        if self.data is None:
            raise ValueError("Archive has no metadata. Cannot extract files.")
        
        df = self.data
        
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} out of range (0-{len(df)-1})")
        
        row = df.iloc[index]
        bin_index = row['bin_index']
        start_offset = row['start_byte_offset']
        file_size = row['file_size_bytes']
        file_format = row.get('format')
        
        # Get the correct bin file
        bin_path = self._get_bin_file_path(bin_index)
        
        if not bin_path.exists():
            raise FileNotFoundError(f"Bin file not found: {bin_path}")
        
        # Read from archive
        with open(bin_path, 'rb') as f:
            f.seek(start_offset)
            audio_bytes = f.read(file_size)
        
        return generic_audio_read(
            audio_bytes,
            file_format,
            start_time,
            end_time
        )
    
    def extract_by_offset(
        self, 
        bin_index: int, 
        start_offset: int, 
        file_size: int,
        file_format: str,
        start_time: int = None,
        end_time: int = None
    ) -> AudioRead:
        """
        Extract a file from the archive by bin index, byte offset and size.
        
        Args:
            bin_index: Index of the bin file
            start_offset: Start byte offset within the bin file
            file_size: File size in bytes
            file_format: Format of the audio file (e.g., 'flac', 'wav', 'mp3')
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
            
        Returns:
            AudioRead object containing the audio data
        """
        bin_path = self._get_bin_file_path(bin_index)
        
        if not bin_path.exists():
            raise FileNotFoundError(f"Bin file not found: {bin_path}")
        
        with open(bin_path, 'rb') as f:
            f.seek(start_offset)
            audio_bytes = f.read(file_size)

        return generic_audio_read(
            audio_bytes,
            file_format,
            start_time,
            end_time
        )
    
    def summary(self):
        """Print a formatted list of files in the archive."""
        if self.data is None:
            print(f"\nArchive: {self.archive_path}")
            print("Archive is empty (no metadata file found).")
            return
        
        df = self.data
        
        print(f"\nArchive: {self.archive_path}")
        print(f"Total files: {len(df)}")

        if len(df) == 0:
            return
        
        # Calculate total size across all bins
        bin_indices = df['bin_index'].unique()
        total_size = 0
        for bin_idx in sorted(bin_indices):
            bin_path = self._get_bin_file_path(bin_idx)
            if bin_path.exists():
                total_size += bin_path.stat().st_size
        
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        print(f"Number of bin files: {len(bin_indices)}")
        
        print("\n" + "="*110)
        for idx, row in df.iterrows():
            filename = Path(row['original_file_path']).name
            size_mb = row['file_size_bytes'] / (1024**2)
            bin_idx = row['bin_index']
            utt_id = row['utt_id']
            # Calculate duration from samples and sample_rate
            duration_seconds = row['length'] / row['sample_rate'] if row['sample_rate'] > 0 else 0
            print(f"{idx:4d} | {utt_id:40s} | Bin{bin_idx} | {filename:40s} | {row['sample_rate']:6d}Hz | "
                    f"{row['channels']:1d}ch | {duration_seconds:7.2f}s | {size_mb:7.2f}MB")

    def clear(self, confirm: bool = False):
        """
        Clear the entire archive (delete all bin files and metadata).
        
        WARNING: This permanently deletes all archive data!
        
        Args:
            confirm: Must be True to actually delete. Safety measure.
        
        Raises:
            ValueError: If confirm is not True
            
        Example:
            archive = Archive('my_archive')
            archive.clear_archive(confirm=True)
        """
        if not confirm:
            raise ValueError(
                "Archive clearing requires confirmation. "
                "Call clear_archive(confirm=True) to proceed. "
                "This will delete all bin files and metadata!"
            )
        
        files_deleted = 0
        
        # Delete metadata file
        if self.metadata_file.exists():
            self.metadata_file.unlink()
            files_deleted += 1
        
        # Delete all bin files
        bin_index = 0
        while True:
            bin_path = self._get_bin_file_path(bin_index)
            if bin_path.exists():
                bin_path.unlink()
                files_deleted += 1
                bin_index += 1
            else:
                break
        
        # Reset data attribute after clearing
        self.data = None
        
        print(f"Archive '{self.archive_path.stem}' cleared successfully!")
        print(f"Deleted {files_deleted} file(s) ({bin_index} bin file(s) + metadata).")
