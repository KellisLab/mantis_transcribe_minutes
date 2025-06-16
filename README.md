# Unified Transcript Processing Platform with Speaker Recognition

A comprehensive Python-based platform for processing meeting recordings, transcripts, and subtitles with advanced speaker recognition capabilities using voice fingerprinting.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Usage Guide](#usage-guide)
  - [Basic Processing](#basic-processing)
  - [Speaker Management](#speaker-management)
  - [Advanced Configuration](#advanced-configuration)
- [Technical Details](#technical-details)
- [API Reference](#api-reference)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

### 🎯 Core Capabilities
- **Multi-Source Input Support**
  - Panopto video URLs and IDs
  - Audio/video files (MP4, MP3, WAV, M4A, etc.)
  - Subtitle files (VTT, SRT, SBV)
  - Excel transcripts
  - Batch directory processing

- **Advanced Speaker Recognition**
  - Voice fingerprinting using ECAPA-TDNN neural networks
  - ChromaDB vector database for speaker embeddings
  - Automatic speaker identification across meetings
  - Confidence scoring and speaker profile management

- **Intelligent Processing Pipeline**
  - WhisperX for speech-to-text transcription
  - Speaker diarization with pyannote
  - NLP-based timestamp refinement
  - AI-powered meeting summarization (GPT-4)

- **Rich Output Formats**
  - Time-synced transcripts with speaker labels
  - Interactive HTML summaries with video links
  - Color-coded Excel files with speaker analytics
  - Detailed speaker participation metrics

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Layer                             │
├─────────────────────────────────────────────────────────────┤
│  Panopto URLs │ Media Files │ Subtitles │ Excel │ Directory │
└────────┬──────┴──────┬──────┴─────┬─────┴───┬───┴─────┬────┘
         │             │            │         │         │
         ▼             ▼            ▼         ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Input Detector & Router                     │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Media Processor     │              │ Subtitle Processor   │
│  - WhisperX          │              │  - Format parsing    │
│  - Diarization       │              │  - Speaker mapping   │
│  - Voice embedding   │              │  - Time alignment    │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                     │
           └──────────────┬──────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Speaker Recognition Engine                      │
│  - SpeechBrain ECAPA-TDNN                                  │
│  - ChromaDB Vector Store                                    │
│  - Similarity Matching                                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Transcript Processor                            │
│  - Format conversion (VTT→TXT→XLSX)                        │
│  - NLP timestamp refinement                                 │
│  - AI summarization (OpenAI GPT-4)                         │
│  - HTML/Markdown generation                                 │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for WhisperX)
- 16GB+ RAM (32GB recommended for large files)
- 10GB+ free disk space for models

### API Keys Required
- **OpenAI API Key**: For meeting summarization
- **Hugging Face Token**: For speaker diarization (get from [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1))

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/transcript-processing-platform.git
cd transcript-processing-platform
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install main requirements
pip install -r requirements.txt

# Install ctranslate2 separately (specific version required)
pip install ctranslate2==4.5.0
```

### 4. Configure Environment
```bash
cp .env.template .env
# Edit .env with your API keys and preferences
```

### 5. Download Models (First Run)
The following models will be downloaded automatically on first use:
- WhisperX large-v2 model (~3GB)
- Pyannote speaker diarization model (~200MB)
- SpeechBrain ECAPA-TDNN model (~100MB)

## Quick Start

### Process Your First Recording
```bash
# Basic processing with speaker recognition
python cli.py meeting_recording.mp4

# View results
ls output/meeting_recording/
```

### Identify and Rename Speakers
```bash
# List identified speakers
python speaker_cli.py list

# Rename generic labels
python speaker_cli.py rename "Speaker_1" "John Doe"
```

## Core Components

### 1. **Input Detector** (`core/input_detector.py`)
Automatically identifies input types using pattern matching and file extension analysis.

### 2. **Pipeline Processor** (`processors/pipeline_processor.py`)
Orchestrates the entire workflow, routing inputs to appropriate processors.

### 3. **Media Processor Enhanced** (`processors/media_processor_enhanced.py`)
- Integrates WhisperX for transcription
- Performs speaker diarization
- Extracts voice embeddings for each speaker segment

### 4. **Speaker Fingerprint Manager** (`speaker_id/fingerprint_manager.py`)
- Manages ChromaDB vector database
- Computes speaker embeddings using ECAPA-TDNN
- Performs similarity matching with configurable thresholds

### 5. **Transcript Processor** (`processors/transcript_processor.py`)
- Converts between formats (VTT → TXT → XLSX)
- Applies NLP-based timestamp refinement
- Generates AI summaries and HTML outputs

## Usage Guide

### Basic Processing

#### Process Video/Audio Files
```bash
# Default: with speaker recognition
python cli.py recording.mp4

# Without speaker recognition (faster)
python cli.py recording.mp4 --no-speaker-recognition

# Custom output directory
python cli.py recording.mp4 --output ./my_transcripts/
```

#### Process Subtitle Files
```bash
# Requires original audio for speaker recognition
python cli.py transcript.srt --audio-path original_recording.mp4

# Without audio (no speaker recognition)
python cli.py transcript.srt
```

#### Process Panopto URLs
```bash
# Note: No speaker recognition available (no audio access)
python cli.py "https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=VIDEO_ID"
```

#### Batch Processing
```bash
# Process entire directory
python cli.py ./recordings/ --speaker-recognition
```

### Speaker Management

#### View Speaker Database
```bash
python speaker_cli.py list
```

#### Rename Speakers
```bash
# By name
python speaker_cli.py rename "Speaker_1" "Alice Johnson"

# By ID prefix
python speaker_cli.py rename "a3f2d8" "Alice Johnson"
```

#### Merge Duplicate Speakers
```bash
# Merge multiple entries into one
python speaker_cli.py merge "John" "J. Doe" "John Doe" --target "John Doe"
```

#### Verify Speaker Identity
```bash
# Check speaker in audio segment
python speaker_cli.py verify audio.wav --start 10 --end 30
```

#### Database Management
```bash
# Export for backup/sharing
python speaker_cli.py export speakers_backup.json

# Import and merge
python speaker_cli.py import speakers_backup.json --merge

# Import and replace
python speaker_cli.py import speakers_backup.json
```

### Advanced Configuration

#### Environment Variables (.env)
```bash
# GPU Configuration
WHISPER_DEVICE=cuda       # or 'cpu' for CPU-only
WHISPER_COMPUTE_TYPE=float16  # or 'int8' for faster/lower quality
WHISPER_BATCH_SIZE=16     # Reduce for less GPU memory

# Model Selection
WHISPER_MODEL=large-v2    # Options: tiny, base, small, medium, large, large-v2

# Processing Settings
BATCH_SIZE_MINUTES=40     # Chunk size for summarization
GPT_MODEL=gpt-4o         # Or gpt-3.5-turbo for cost savings

# Directories
OUTPUT_DIR=./output
SPEAKER_DB_DIR=./speaker_database
```

#### Custom Speaker Database
```bash
# Use shared/team database
python cli.py recording.mp4 --speaker-db-path /shared/team_speakers/
```

#### Confidence Threshold Adjustment
Edit `speaker_id/fingerprint_manager.py`:
```python
def __init__(self, ..., similarity_threshold: float = 0.85):
    # 0.85 = 85% similarity required
    # Lower values: more lenient matching
    # Higher values: stricter matching
```

## Technical Details

### Speaker Embedding Architecture
The system uses **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network):
- 192-dimensional speaker embeddings
- Trained on VoxCeleb dataset
- Cosine similarity for speaker matching
- ChromaDB for efficient vector search

### Voice Fingerprinting Process
1. **Audio Segmentation**: Extract speaker-specific segments from diarization
2. **Feature Extraction**: Convert audio to 192-dim embedding vectors
3. **Vector Storage**: Store in ChromaDB with metadata
4. **Similarity Search**: Compare new embeddings against database
5. **Identity Resolution**: Apply threshold and confidence scoring

### Transcript Enhancement Pipeline
1. **Format Normalization**: Convert all inputs to standardized VTT
2. **Timestamp Refinement**: Use NLTK for sentence boundary detection
3. **Speaker Attribution**: Map diarization results to transcript segments
4. **Topic Detection**: Identify topic changes using TF-IDF
5. **AI Summarization**: Generate hierarchical summaries with GPT-4

## API Reference

### Pipeline Processor
```python
from processors.pipeline_processor import PipelineProcessor

# Initialize pipeline
pipeline = PipelineProcessor(
    output_base_dir=Path("./output"),
    enable_speaker_recognition=True
)

# Process media file
result = pipeline.process(
    "meeting.mp4",
    language="en",
    speaker_recognition=True,
    auto_add_speakers=True,
    skip_refinement=False
)

# Access results
print(result['success'])          # Processing status
print(result['output_dir'])       # Output location
print(result['speakers'])         # Identified speakers
print(result['files'])           # Generated files
```

### Speaker Manager
```python
from speaker_id.fingerprint_manager import SpeakerFingerprintManager

# Initialize manager
manager = SpeakerFingerprintManager(
    db_path=Path("./speaker_db"),
    similarity_threshold=0.85
)

# Extract embedding from audio
embedding = manager.extract_embedding(
    "audio.wav",
    start_time=10.0,
    end_time=20.0
)

# Identify speaker
result = manager.identify_speaker(
    embedding,
    return_all_matches=True
)

# Add new speaker
speaker_id = manager.add_speaker(
    name="John Doe",
    embedding=embedding,
    metadata={"department": "Engineering"}
)
```

## Performance Optimization

### GPU Memory Management
```bash
# Reduce batch size for limited GPU memory
WHISPER_BATCH_SIZE=8  # Default: 16

# Use INT8 quantization
WHISPER_COMPUTE_TYPE=int8  # Faster, slightly lower quality
```

### Processing Speed
- **CPU-only**: ~5x slower than GPU
- **Small model**: ~10x faster than large-v2
- **Skip diarization**: ~2x faster
- **Skip refinement**: ~1.5x faster

### Batch Processing Tips
```bash
# Process overnight with logging
nohup python cli.py ./recordings/ > processing.log 2>&1 &

# Monitor progress
tail -f processing.log
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
export WHISPER_BATCH_SIZE=4

# Solution 2: Use CPU
export WHISPER_DEVICE=cpu

# Solution 3: Use smaller model
export WHISPER_MODEL=base
```

#### Low Speaker Confidence
- Ensure audio quality is good (low noise, clear speech)
- Use segments longer than 3 seconds
- Process multiple recordings to build better profiles
- Adjust similarity threshold if needed

#### Import Errors
```bash
# Reinstall with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install missing dependencies
pip install speechbrain chromadb scipy nltk
```

#### Panopto Access Issues
- Ensure you have proper permissions
- Check if video is public/unlisted
- Verify URL format is correct

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install dev dependencies
pip install pytest black flake8 mypy
```

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings for all public methods
- Run `black .` before committing

### Testing
```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=processors tests/
```

### Pull Request Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **WhisperX**: Faster Whisper transcription with word-level timestamps
- **Pyannote**: State-of-the-art speaker diarization
- **SpeechBrain**: ECAPA-TDNN speaker recognition model
- **ChromaDB**: Efficient vector database for embeddings
- **OpenAI**: GPT-4 for intelligent summarization

## Citation

If you use this platform in your research, please cite:
```bibtex
@software{unified_transcript_platform,
  title = {Unified Transcript Processing Platform with Speaker Recognition},
  year = {2025},
  url = {https://github.com/yourusername/transcript-processing-platform}
}
```