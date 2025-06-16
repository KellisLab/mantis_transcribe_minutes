# File: processors/media_processor.py
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from processors.base_processor import BaseProcessor
from core.config import Config

# Only import whisperx if available
try:
    import whisperx
    import torch
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("Warning: WhisperX not installed. Media transcription will not be available.")

class MediaProcessor(BaseProcessor):
    """Processor for transcribing audio/video files using WhisperX."""
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.model = None
        self.device = Config.WHISPER_DEVICE
        self.compute_type = Config.WHISPER_COMPUTE_TYPE
        self.model_size = Config.WHISPER_MODEL
        self.batch_size = Config.WHISPER_BATCH_SIZE
    
    def process(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process media file to generate transcript.
        
        Args:
            input_path: Path to media file
            **kwargs: Additional options
        
        Returns:
            Dict with processing results
        """
        if not WHISPERX_AVAILABLE:
            return {
                'success': False,
                'error': 'WhisperX is not installed. Run: pip install whisperx'
            }
        
        # Check if HF token is available for diarization
        if not Config.HF_TOKEN:
            print("Warning: HF_TOKEN not set. Speaker diarization will be skipped.")
        
        input_file = Path(input_path)
        if not input_file.exists():
            return {
                'success': False,
                'error': f'Media file not found: {input_path}'
            }
        
        try:
            # Load model
            print(f"Loading WhisperX model '{self.model_size}' on {self.device}...")
            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                download_root=str(Config.MODELS_DIR),
                language=kwargs.get('language', 'en')
            )
            
            # Load audio
            print(f"Loading audio from '{input_file}'...")
            audio = whisperx.load_audio(str(input_file))
            
            # Transcribe
            print("Running transcription...")
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            
            # Force alignment
            print("Running alignment...")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            
            # Speaker diarization if HF token available
            if Config.HF_TOKEN:
                print("Running speaker diarization...")
                from whisperx.diarize import DiarizationPipeline
                
                diarize_model = DiarizationPipeline(
                    use_auth_token=Config.HF_TOKEN,
                    device=self.device,
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Save transcript as srt
            srt_file = self._save_transcript(result, input_file.stem)
            
            # Save metadata
            metadata = {
                'source_file': str(input_file),
                'duration': result.get('duration', 0),
                'language': result.get('language', 'unknown'),
                'segments_count': len(result.get('segments', [])),
                'transcript_file': str(srt_file),
                'has_speakers': 'speaker' in str(result.get('segments', [{}])[0])
            }
            self.save_metadata(metadata)
            
            return {
                'success': True,
                'transcript_file': srt_file,
                'metadata': metadata,
                'segments': result.get('segments', [])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Transcription failed: {str(e)}'
            }
        finally:
            # Clean up GPU memory if used
            if self.model and self.device == 'cuda':
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    
    def _save_transcript(self, result: Dict, base_name: str) -> Path:
        """Save transcript in srt format."""
        from whisperx.utils import get_writer
        
        # Prepare for srt output
        output_format = "srt"
        options = {
            "highlight_words": False,
            "max_line_width": None,
            "max_line_count": None
        }
        
        # Ensure language key exists
        result["language"] = result.get("language", "en")
        
        # Write srt file
        # writer = get_writer(output_format, str(self.output_dir))
        writer = get_writer("srt", str(self.output_dir))
        
        # Create a mock audio path for the writer
        mock_audio_path = str(self.output_dir / f"{base_name}.tmp")
        writer(result, mock_audio_path, options)
        
        # The writer creates a file with the audio file's base name
        srt_file = self.output_dir / f"{base_name}.srt"
        
        print(f"Transcript saved to: {srt_file}")
        return srt_file