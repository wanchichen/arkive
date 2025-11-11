#!/usr/bin/env python3
"""
Audio Archive Tool
Converts mixed audio formats to 16-bit PCM FLAC and stores in a single archive with Parquet metadata.
"""

import io
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import subprocess
import tempfile
from tqdm import tqdm

from arkive.audio_read import generic_audio_read
from arkive.definitions import AudioRead

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
        flush_interval: int = 10
    ):
        """
        Create a new archive from a list of audio files.
        
        Args:
            audio_files: List of paths to audio files (WAV, FLAC, MP3, OPUS, etc.)
            target_format: Target format for conversion ('flac', 'wav', 'mp3', 'opus', or None to retain original)
            show_progress: Whether to show progress messages
            target_bit_depth: Target bit depth for conversion (16, 32, or 64). Only applies when target_format is not None.
            flush_interval: Flush bin file buffer to disk every N files (default: 10). Set to 0 to disable periodic flushing.
        """
        metadata_records = []
        current_bin_index, current_offset = self._get_current_bin_info()
        
        # Validate parameters
        if target_format is not None:
            target_format = target_format.lower()
            valid_formats = ['flac', 'wav', 'mp3', 'opus']
            if target_format not in valid_formats:
                raise ValueError(f"target_format must be one of {valid_formats} or None, got '{target_format}'")
        
        if target_bit_depth not in [16, 32, 64]:
            raise ValueError(f"target_bit_depth must be 16, 32, or 64, got {target_bit_depth}")
        
        # Open the current bin file for appending
        current_bin_path = self._get_bin_file_path(current_bin_index)
        current_bin_file = open(current_bin_path, 'ab')
        
        try:
            for i, audio_file in tqdm(enumerate(audio_files)):
                try:
                    orig_format = audio_file.split('.')[-1].lower()
                    
                    # First, get original file info including bit depth
                    orig_file_data, orig_audio_data, orig_sample_rate, orig_channels, orig_samples, orig_format, orig_bit_depth = self._read_original_format(audio_file)
                    
                    # Determine if conversion is needed
                    needs_conversion = False
                    if target_format is not None:
                        # Check if format or bit depth differs
                        if orig_format != target_format or orig_bit_depth != target_bit_depth:
                            needs_conversion = True
                    
                    if needs_conversion:
                        # Convert to target format and bit depth
                        file_data, sample_rate, channels, samples = self._convert_to_format(
                            orig_audio_data, orig_sample_rate, orig_channels, target_format, target_bit_depth
                        )
                        file_format = target_format
                        bit_depth = target_bit_depth
                    else:
                        # Use original format and bit depth
                        file_data = orig_file_data
                        sample_rate = orig_sample_rate
                        channels = orig_channels
                        samples = orig_samples
                        file_format = orig_format
                        bit_depth = orig_bit_depth
                    
                    file_size = len(file_data)
                    
                    # Check if we need to create a new bin file
                    if current_offset + file_size > self.MAX_BIN_SIZE:
                        # Close current bin file
                        current_bin_file.close()
                        
                        # Move to next bin file
                        current_bin_index += 1
                        current_offset = 0
                        current_bin_path = self._get_bin_file_path(current_bin_index)
                        current_bin_file = open(current_bin_path, 'wb')
                        
                        if show_progress:
                            print(f"Creating new bin file: {current_bin_path.name} (previous bin full)")
                    
                    # Write to current bin file
                    current_bin_file.write(file_data)
                    
                    # Record metadata
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
                    
                    # Periodically flush buffer to disk to ensure data persistence
                    # Note: fsync() ensures data is written to disk, but adds ~1-10ms overhead per call
                    # Default flush_interval=10 provides good balance between safety and performance
                    if flush_interval > 0 and (i + 1) % flush_interval == 0:
                        try:
                            current_bin_file.flush()
                            os.fsync(current_bin_file.fileno())  # Force OS-level write to disk
                            if show_progress:
                                print(f"Flushed buffer to disk (processed {i + 1} files)")
                        except (OSError, IOError) as e:
                            # Log warning but continue processing
                            print(f"Warning: Failed to flush data to disk: {e}")
                            # Data is still in OS buffer and will be written eventually
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        finally:
            # Final flush before closing to ensure all data is written
            if flush_interval > 0:
                try:
                    current_bin_file.flush()
                    os.fsync(current_bin_file.fileno())
                except (OSError, IOError) as e:
                    print(f"Warning: Failed to flush final data to disk: {e}")
            current_bin_file.close()
        
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
            print(f"Total files: {len(metadata_records)}")
            
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
    ) -> AudioRead:
        """
        Extract a file from the archive by bin index, byte offset and size.
        
        Args:
            bin_index: Index of the bin file
            start_offset: Start byte offset within the bin file
            file_size: File size in bytes
            output_path: Optional path to save the extracted file
            
        Returns:
            Binary data of the FLAC file
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