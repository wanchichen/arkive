#!/usr/bin/env python3
"""
Audio Archive Tool
Converts mixed audio formats to 16-bit PCM FLAC and stores in a single archive with Parquet metadata.
"""

import io
import os
from pathlib import Path
from typing import List, Optional, Tuple
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


# Static function for multiprocessing (must be at module level for pickle)
def _process_single_audio_file_static(
    audio_file: str, target_format: Optional[str], target_bit_depth: int
) -> Optional[Tuple]:
    """
    Static version of _process_single_audio_file for multiprocessing.
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
                # Get bit depth
                subtype_upper = info.subtype.upper()
                if 'PCM_16' in subtype_upper or 'PCM16' in subtype_upper:
                    orig_bit_depth = 16
                elif 'PCM_24' in subtype_upper or 'PCM24' in subtype_upper:
                    orig_bit_depth = 24
                elif 'PCM_32' in subtype_upper or 'PCM32' in subtype_upper:
                    orig_bit_depth = 32
                elif 'FLOAT' in subtype_upper and 'DOUBLE' not in subtype_upper:
                    orig_bit_depth = 32
                elif 'DOUBLE' in subtype_upper:
                    orig_bit_depth = 64
                else:
                    orig_bit_depth = 16
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
            # Convert
            if channels == 1:
                audio_data = audio_data.reshape(-1, 1)
            
            if target_bit_depth == 16:
                subtype = 'PCM_16'
            elif target_bit_depth == 32:
                subtype = 'PCM_32'
            elif target_bit_depth == 64:
                subtype = 'DOUBLE'
            
            out_buf = io.BytesIO()
            out_buf.name = f'temp.{target_format}'
            sf.write(out_buf, audio_data, sample_rate, 
                    subtype=subtype, format=target_format.upper())
            out_buf.seek(0)
            file_data = out_buf.read()
            
            file_format = target_format
            bit_depth = target_bit_depth
            samples = len(audio_data)
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
            
            # Get bit depth
            subtype_upper = info.subtype.upper()
            if 'PCM_16' in subtype_upper or 'PCM16' in subtype_upper:
                orig_bit_depth = 16
            elif 'PCM_24' in subtype_upper or 'PCM24' in subtype_upper:
                orig_bit_depth = 24
            elif 'PCM_32' in subtype_upper or 'PCM32' in subtype_upper:
                orig_bit_depth = 32
            elif 'FLOAT' in subtype_upper and 'DOUBLE' not in subtype_upper:
                orig_bit_depth = 32
            elif 'DOUBLE' in subtype_upper:
                orig_bit_depth = 64
            else:
                orig_bit_depth = 16
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
            # Convert
            if channels == 1:
                audio_data = audio_data.reshape(-1, 1)
            
            if target_bit_depth == 16:
                subtype = 'PCM_16'
            elif target_bit_depth == 32:
                subtype = 'PCM_32'
            elif target_bit_depth == 64:
                subtype = 'DOUBLE'
            
            out_buf = io.BytesIO()
            out_buf.name = f'temp.{target_format}'
            sf.write(out_buf, audio_data, sample_rate, 
                    subtype=subtype, format=target_format.upper())
            out_buf.seek(0)
            file_data = out_buf.read()
            
            file_format = target_format
            bit_depth = target_bit_depth
            samples = len(audio_data)
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


class Arkive:
    """Create and manage audio archives with mixed format support."""
    
    MAGIC_NUMBER = b'AUDARCH1'
    MAX_BIN_SIZE = 32 * 1024 * 1024 * 1024  # default 32 GB in bytes
    
    def __init__(self, archive_dir: str):
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

    def append(
        self, 
        audio_files: List[str],
        target_format: Optional[str] = 'flac', 
        show_progress: bool = False,
        target_bit_depth: int = 16,
        flush_interval: int = 10,
        num_workers: int = None
    ):
        """
        Create a new archive from a list of audio files (with multiprocessing support).
        
        Args:
            audio_files: List of paths to audio files (WAV, FLAC, MP3, OPUS, etc.)
            target_format: Target format for conversion ('flac', 'wav', 'mp3', 'opus', or None to retain original)
            show_progress: Whether to show progress messages
            target_bit_depth: Target bit depth for conversion (16, 32, or 64). Only applies when target_format is not None.
            flush_interval: Flush bin file buffer to disk every N files (default: 10). Set to 0 to disable periodic flushing.
            num_workers: Number of worker processes for parallel processing (default: cpu_count() - 1, set to 1 to disable)
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
            # Use multiprocessing if num_workers > 1
            if num_workers > 1:
                pool = Pool(processes=num_workers)
                process_func = partial(
                    _process_single_audio_file_static,
                    target_format=target_format,
                    target_bit_depth=target_bit_depth
                )
                
                # Process files in parallel
                # imap_unordered is faster than imap
                # Since we do not consider the order of audio paths, we can write
                # any audio file if it is available.
                results_iter = pool.imap_unordered(process_func, audio_files)
                if show_progress:
                    results_iter = tqdm(results_iter, total=len(audio_files), desc="Processing audio files")
                
                # NOTE(qingzheng): we cannot do parallel write into *single .bin* file, as:
                # 1. we need to keep the order of current_offset
                # 2. there is a system lock for the .bin file write
                for i, result in enumerate(results_iter):
                    if result is None:
                        continue
                    
                    audio_file, file_data, sample_rate, channels, samples, file_format, bit_depth = result
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
                    
                    # Write to bin file and record metadata
                    current_bin_file.write(file_data)
                    metadata_records.append({
                        'original_file_path': str(Path(audio_file).absolute()),
                        'bin_index': current_bin_index,
                        'path': str(current_bin_path),
                        'start_byte_offset': current_offset,
                        'file_size_bytes': file_size,
                        'sample_rate': sample_rate,
                        'channels': channels,
                        'length': samples,
                        'format': file_format,
                        'bit_depth': bit_depth
                    })
                    current_offset += file_size
                    processed_count += 1
                    
                    # Periodic flush
                    if flush_interval > 0 and (i + 1) % flush_interval == 0:
                        self._flush_data(current_bin_file, metadata_records, show_progress, i + 1)
                        metadata_records = []
                
                pool.close()
                pool.join()
            
            else:
                # Single process mode (original logic)
                iter_obj = tqdm(enumerate(audio_files), total=len(audio_files), desc="Processing audio files") if show_progress else enumerate(audio_files)
                for i, audio_file in iter_obj:
                    try:
                        result = self._process_single_audio_file(audio_file, target_format, target_bit_depth)
                        if result is None:
                            continue
                        
                        _, file_data, sample_rate, channels, samples, file_format, bit_depth = result
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
                        
                        # Write to bin file and record metadata
                        current_bin_file.write(file_data)
                        metadata_records.append({
                            'original_file_path': str(Path(audio_file).absolute()),
                            'bin_index': current_bin_index,
                            'path': str(current_bin_path),
                            'start_byte_offset': current_offset,
                            'file_size_bytes': file_size,
                            'sample_rate': sample_rate,
                            'channels': channels,
                            'length': samples,
                            'format': file_format,
                            'bit_depth': bit_depth
                        })
                        current_offset += file_size
                        processed_count += 1
                        
                        # Periodic flush
                        if flush_interval > 0 and (i + 1) % flush_interval == 0:
                            self._flush_data(current_bin_file, metadata_records, show_progress, i + 1)
                            metadata_records = []
                    
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
                        continue
        
        finally:
            # Final flush before closing to ensure all data is written
            if flush_interval > 0 and metadata_records:
                self._flush_data(current_bin_file, metadata_records, show_progress, processed_count)
                metadata_records = []
            current_bin_file.close()
        
        if not flush_interval > 0:
            # Append to existing metadata or create new
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
        # Validate parameters
        if target_format is not None:
            target_format = target_format.lower()
            valid_formats = ['flac', 'wav', 'mp3', 'opus']
            if target_format not in valid_formats:
                raise ValueError(f"target_format must be one of {valid_formats} or None")
        
        if target_bit_depth not in [16, 32, 64]:
            raise ValueError(f"target_bit_depth must be 16, 32, or 64, got {target_bit_depth}")
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        # Get current bin info
        current_bin_index, current_offset = self._get_current_bin_info()
        
        # Open bin file for appending
        current_bin_path = self._get_bin_file_path(current_bin_index)
        current_bin_file = open(current_bin_path, 'ab')
        
        metadata_records = []
        processed_count = 0
        
        try:
            # Process in batches
            batch = []

            if show_progress:
                audio_bytes_iterator = tqdm(audio_bytes_iterator, desc="Processing audios")
            
            for audio_item in audio_bytes_iterator:
                batch.append(audio_item)
                
                # When batch is full, process it
                if len(batch) >= batch_size:
                    current_bin_file, current_bin_index, current_offset, processed_count, metadata_records = \
                        self._process_bytes_batch(
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
                            flush_interval
                        )
                    batch = []  # Clear batch to free memory
            
            # Process remaining batch
            if batch:
                current_bin_file, current_bin_index, current_offset, processed_count, metadata_records = \
                    self._process_bytes_batch(
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
                        flush_interval
                    )
        
        finally:
            # Final flush
            if flush_interval > 0 and metadata_records:
                self._flush_data(current_bin_file, metadata_records, show_progress, processed_count)
                metadata_records = []
            current_bin_file.close()
        
        # Save remaining metadata if not using flush
        if not flush_interval > 0:
            new_df = pd.DataFrame(metadata_records)
            
            if self.metadata_file.exists():
                existing_df = self.data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                self.data = combined_df
                combined_df.to_parquet(self.metadata_file, index=False)
                
                if show_progress:
                    print(f"\nAppended {len(metadata_records)} files to existing archive!")
            else:
                new_df.to_parquet(self.metadata_file, index=False)
                self.data = new_df
        
        if show_progress:
            print(f"\nArchive created successfully!")
            print(f"Total files processed: {processed_count}")
    
    def _process_bytes_batch(
        self,
        batch: List[dict],
        current_bin_file,
        current_bin_index: int,
        current_offset: int,
        processed_count: int,
        metadata_records: List[dict],
        target_format: Optional[str],
        target_bit_depth: int,
        num_workers: int,
        show_progress: bool,
        flush_interval: int
    ):
        """Process a batch of audio bytes with parallel processing."""
        
        # Create process pool for this batch
        pool = Pool(processes=num_workers)
        process_func = partial(
            _process_audio_from_bytes_static,
            target_format=target_format,
            target_bit_depth=target_bit_depth
        )
        
        # Process batch in parallel (use imap_unordered to avoid bubble)
        results_iter = pool.imap_unordered(process_func, batch)
        
        for result in results_iter:
            if result is None:
                continue
            
            audio_key, file_data, sample_rate, channels, samples, file_format, bit_depth = result
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
            
            # Write to bin file and record metadata
            current_bin_file.write(file_data)
            metadata_records.append({
                'original_file_path': audio_key,  # Use key as identifier
                'bin_index': current_bin_index,
                'path': str(self._get_bin_file_path(current_bin_index)),
                'start_byte_offset': current_offset,
                'file_size_bytes': file_size,
                'sample_rate': sample_rate,
                'channels': channels,
                'length': samples,
                'format': file_format,
                'bit_depth': bit_depth
            })
            current_offset += file_size
            processed_count += 1
            
            # Periodic flush
            if flush_interval > 0 and processed_count % flush_interval == 0:
                self._flush_data(current_bin_file, metadata_records, show_progress, processed_count)
                metadata_records = []
        
        pool.close()
        pool.join()
        
        return current_bin_file, current_bin_index, current_offset, processed_count, metadata_records
    
    def _process_single_audio_file(self, audio_file: str, target_format: Optional[str], 
                                    target_bit_depth: int) -> Optional[Tuple]:
        """
        Process a single audio file (read and convert if needed).
        
        Args:
            audio_file: Path to audio file
            target_format: Target format for conversion
            target_bit_depth: Target bit depth
            
        Returns:
            Tuple of (audio_file, file_data, sample_rate, channels, samples, format, bit_depth) or None if failed
        """
        try:
            # Read original format
            orig_file_data, orig_audio_data, orig_sample_rate, orig_channels, \
                orig_samples, orig_format, orig_bit_depth = self._read_original_format(audio_file)
            
            # Determine if conversion is needed
            needs_conversion = False
            if target_format is not None:
                if orig_format != target_format or orig_bit_depth != target_bit_depth:
                    needs_conversion = True
            
            if needs_conversion:
                file_data, sample_rate, channels, samples = self._convert_to_format(
                    orig_audio_data, orig_sample_rate, orig_channels, target_format, target_bit_depth
                )
                file_format = target_format
                bit_depth = target_bit_depth
            else:
                file_data = orig_file_data
                sample_rate = orig_sample_rate
                channels = orig_channels
                samples = orig_samples
                file_format = orig_format
                bit_depth = orig_bit_depth
            
            return (audio_file, file_data, sample_rate, channels, samples, file_format, bit_depth)
        
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return None
    
    def _flush_data(self, current_bin_file, metadata_records: List[dict], 
                    show_progress: bool, file_count: int):
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
    
    def _get_bit_depth_from_subtype(self, subtype: str) -> int:
        """
        Get bit depth from soundfile subtype string.
        
        Args:
            subtype: Subtype string from soundfile (e.g., 'PCM_16', 'PCM_24', 'FLOAT', 'DOUBLE')
            
        Returns:
            Bit depth as integer
        """
        subtype_upper = subtype.upper()
        
        # PCM formats
        if 'PCM_16' in subtype_upper or 'PCM16' in subtype_upper:
            return 16
        elif 'PCM_24' in subtype_upper or 'PCM24' in subtype_upper:
            return 24
        elif 'PCM_32' in subtype_upper or 'PCM32' in subtype_upper:
            return 32
        elif 'PCM_8' in subtype_upper or 'PCM8' in subtype_upper:
            return 8
        # Float formats
        elif 'FLOAT' in subtype_upper and 'DOUBLE' not in subtype_upper:
            return 32  # Single precision float
        elif 'DOUBLE' in subtype_upper:
            return 64  # Double precision float
        # Default
        else:
            return 16  # Default to 16-bit if unknown
    
    def _convert_to_format(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        channels: int,
        target_format: str,
        target_bit_depth: int
    ) -> tuple:
        """
        Convert audio file to specified format and bit depth.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz
            channels: Number of channels (will be recalculated from audio_data if inconsistent)
            target_format: Target format ('flac', 'wav', 'mp3', 'opus')
            target_bit_depth: Bit depth (16, 32, or 64)
            
        Returns:
            Tuple of (file_data, sample_rate, channels, samples)
        """
        
        # Get number of channels from audio data (recalculate to ensure consistency)
        if channels == 1:
            audio_data = audio_data.reshape(-1, 1)
        
        # Convert to target bit depth
        if target_bit_depth == 16:
            subtype = 'PCM_16'
        elif target_bit_depth == 32:
            subtype = 'PCM_32'
        elif target_bit_depth == 64:
            subtype = 'DOUBLE'  # 64-bit float for FLAC/WAV
        
        if target_format in ['flac', 'wav']:
            # Use soundfile for FLAC and WAV
            # tempfile.SpooledTemporaryFile may be a better solution
            out_buf = io.BytesIO()
            out_buf.name = f'temp.{target_format}'
            sf.write(out_buf, audio_data, sample_rate, 
                    subtype=subtype, format=target_format.upper())
            out_buf.seek(0)
            audio_binary = out_buf.read()
        elif target_format in ['mp3', 'opus']:
            # Use ffmpeg for MP3 and OPUS
            raise NotImplementedError("Not yet validated")
            #with tempfile.NamedTemporaryFile(suffix=f'.{target_format}', delete=False) as tmp_file:
            #    tmp_path = tmp_file.name
            #self._write_with_ffmpeg(audio_data, sample_rate, tmp_path, target_format, target_bit_depth)
        
            # Read the file as binary
            #with open(tmp_path, 'rb') as f:
            #    audio_binary = f.read()
        
        return audio_binary, sample_rate, channels, len(audio_data)
    
    def _read_original_format(self, audio_file: str) -> tuple:
        """
        Read audio file in its original format without conversion.
        
        Args:
            audio_file: Path to input audio file
            
        Returns:
            Tuple of (file_data, sample_rate, channels, duration, format, bit_depth)
        """
        # Read the file as binary
        if ':' in audio_file and 'ark' in audio_file.lower():
            # Temporarily used for OWSM data
            try:
                import kaldiio
            except ImportError:
                raise ImportError("kaldiio is not installed.")
            sample_rate, audio_data = kaldiio.load_mat(audio_file)

            channels = 1
            bit_depth = 16
            file_format = 'flac'
            buffer = io.BytesIO()
            buffer.name = f'temp.flac'
            sf.write(buffer, audio_data, sample_rate, format='FLAC', subtype='PCM_16')
            buffer.seek(0)
            file_data = buffer.read()
        else:
            with open(audio_file, 'rb') as f:
                file_data = f.read()
        
            # Get audio info for metadata
            try:
                audio_data, sample_rate = sf.read(audio_file, dtype='float32')
                # Get bit depth from file info
                info = sf.info(audio_file)
                bit_depth = self._get_bit_depth_from_subtype(info.subtype)
            except:
                # Fall back to ffmpeg for formats not supported by soundfile
                audio_data, sample_rate = self._read_with_ffmpeg(audio_file)
                bit_depth = 16  # Default for ffmpeg conversion
            
            # Get number of channels
            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[1]
            
            # Determine format from file extension
            file_format = Path(audio_file).suffix.lower().lstrip('.')
        
        return file_data, audio_data, sample_rate, channels, len(audio_data), file_format, bit_depth
    
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
            # Calculate duration from samples and sample_rate
            duration_seconds = row['length'] / row['sample_rate'] if row['sample_rate'] > 0 else 0
            print(f"{idx:4d} | Bin{bin_idx} | {filename:40s} | {row['sample_rate']:6d}Hz | "
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