# arkive

## Installation

```bash
git clone https://github.com/wanchichen/arkive.git
cd arkive
pip install -e .
```

**Additional dependency**: FFmpeg must be installed for MP3/OPUS/M4A support:
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

### Creating a new archive file
```python
from arkive import Arkive
test = Arkive('test_ark')
```

Will produce two files:

- `test_ark.bin` is the binarized audio
- `test_ark.parquet` is the metadata table

If these files already exist, `test = Arkive('test_ark')` will perform a read.

### Adding files to archive
```python
test.append(['audio.wav', 'audio.flac'])
test.append(['audio.wav', 'audio.flac'], target_format="wav") # defaults to flac
test.append(['audio.wav', 'audio.flac'], bit_depth=32) # defaults to 16-bit PCM
```

If `test_ark.bin` overflows (grows beyond 32GB), it will automatically create and manage additional binary dumps to prevent gigantic files.

### Viewing data
```python
>>> test.data
                                  original_file_path  bin_index  start_byte_offset  file_size_bytes  sample_rate  channels  duration_seconds format  bit_depth
0  /work/nvme/bbjs/chen26/data_hub/tvseries/downl...          0                  0         66368761        16000         1       4616.528625   flac         16
1  /work/nvme/bbjs/chen26/data_hub/tvseries/downl...          0           66368761         66368761        16000         1       4616.528625   flac         16

>>> test.summary()

Archive: /work/nvme/bbjs/chen26/test_ark
Total files: 2
Total size: 0.12 GB
Number of bin files: 1

==============================================================================================================
   0 | Bin0 | American_Love_Story_Full_Movie-[4XXDw_PbLJE].wav |  16000Hz | 1ch | 4616.53s |   63.29MB
   1 | Bin0 | American_Love_Story_Full_Movie-[4XXDw_PbLJE].wav |  16000Hz | 1ch | 4616.53s |   63.29MB
```

### Reading files from initialized archive

```python
>>> test.extract_file(index=1)
AudioRead(file_type='flac', modality='audio', sample_rate=16000, array=array([[0.00030518],
       [0.00042725],
       [0.0005188 ],
       ...,
       [0.00039673],
       [0.0065918 ],
       [0.00039673]]))
>>> test.extract_file(index=1, start_time=25, end_time=100)
AudioRead(file_type='flac', modality='audio', sample_rate=16000, array=array([[0.00030518],
       [0.00042725],
       [0.0005188 ],
       ...,
       [0.00039673],
       [0.0065918 ],
       [0.00039673]]))
```
### Reading files directly from archive with metadata

```python
>>> arkive.audio_read('/work/nvme/bbjs/chen26/test_ark.bin', start_offset=66368761, file_size=66368761, start_time=0, end_time=None)
AudioRead(file_type='flac', modality='audio', sample_rate=16000, array=array([[0.00030518],
       [0.00042725],
       [0.0005188 ],
       ...,
       [0.02752686],
       [0.0289917 ],
       [0.02993774]]))
```

Data stored as `.wav` files as the underlying typing support random access partial reads, allowing you to read in a specific timespan directly into memory without loading the full file. For all other data types, the full file is read and then segmented if `start_time` and/or `end_time` are used.

### Reading files from remote archive with metadata

```python
arkive.audio_read('https://my_url.com/test_ark_small.bin', 0, 66368761)
AudioRead(file_type='flac', modality='audio', sample_rate=16000, array=array([[0.00030518],
       [0.00042725],
       [0.0005188 ],
       ...,
       [0.02752686],
       [0.0289917 ],
       [0.02993774]]))
```

### Deleting an archive from Python

```python
test.clear(confirm=True)
```
