# File: processors/pipeline_processor_updated.py
"""
Updated Pipeline Processor with integrated speaker recognition.
This replaces the original pipeline_processor.py to use enhanced processors.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.config import Config
from core.input_detector import InputDetector, InputType
from core.utils import sanitize_filename

# Import enhanced processors with speaker recognition
from processors.panopto_processor import PanoptoProcessor
from processors.media_processor_enhanced import EnhancedMediaProcessor
from processors.subtitle_processor_enhanced import EnhancedSubtitleProcessor
from processors.transcript_processor import TranscriptProcessor


class PipelineProcessor:
    """Main pipeline that orchestrates the entire processing workflow with speaker recognition."""
    
    def __init__(self, output_base_dir: Optional[Path] = None, 
                 enable_speaker_recognition: bool = True):
        """
        Initialize the pipeline processor.
        
        Args:
            output_base_dir: Base directory for outputs
            enable_speaker_recognition: Enable speaker fingerprinting
        """
        self.output_base_dir = output_base_dir or Config.OUTPUT_DIR
        self.current_output_dir = None
        self.meeting_name = None
        self.meeting_metadata = {}
        self.enable_speaker_recognition = enable_speaker_recognition
    
    def process(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process any supported input through the appropriate pipeline.
        
        Additional kwargs:
            - speaker_recognition: Override default speaker recognition setting
            - auto_add_speakers: Automatically add unknown speakers (default: True)
            - speaker_db_path: Custom path for speaker database
        """
        # Check if speaker recognition is requested
        use_recognition = kwargs.get('speaker_recognition', self.enable_speaker_recognition)
        
        # Detect input type
        input_type, metadata = InputDetector.detect(input_path)
        
        if input_type == InputType.UNKNOWN:
            return {
                'success': False,
                'error': f'Unable to determine input type for: {input_path}'
            }
        
        print(f"\nProcessing {input_type.value}: {input_path}")
        if use_recognition:
            print("Speaker recognition: ENABLED")
        
        # Route to appropriate processor
        if input_type in [InputType.PANOPTO_URL, InputType.PANOPTO_ID]:
            return self._process_panopto(input_path, metadata, **kwargs)
        
        elif input_type == InputType.MEDIA_FILE:
            return self._process_media(input_path, **kwargs)
        
        elif input_type == InputType.SUBTITLE_FILE:
            return self._process_subtitle(input_path, **kwargs)
        
        elif input_type == InputType.EXCEL_FILE:
            return self._process_excel(input_path, **kwargs)
        
        elif input_type == InputType.DIRECTORY:
            return self._process_directory(input_path, **kwargs)
        
        else:
            return {
                'success': False,
                'error': f'Processing not implemented for type: {input_type.value}'
            }
    
    def _get_unique_directory_name(self, base_dir: Path, folder_name: str) -> str:
        """Generate a unique directory name by appending a suffix if necessary."""
        dir_path = base_dir / folder_name
        
        if dir_path.exists() and any(dir_path.iterdir()):
            counter = 2
            while True:
                new_name = f"{folder_name} ({counter})"
                new_path = base_dir / new_name
                
                if not new_path.exists() or not any(new_path.iterdir()):
                    return new_name
                
                counter += 1
        
        return folder_name
    
    def _create_output_directory(self, meeting_name: str, metadata: Dict[str, Any]) -> Path:
        """Create output directory following original naming convention."""
        safe_name = sanitize_filename(meeting_name)
        unique_name = self._get_unique_directory_name(self.output_base_dir, safe_name)
        
        output_dir = self.output_base_dir / unique_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_output_dir = output_dir
        self.meeting_name = meeting_name
        self.meeting_metadata = metadata
        
        print(f"Creating output directory: {output_dir}")
        
        return output_dir
    
    def _process_panopto(self, input_path: str, video_id: Optional[str], **kwargs) -> Dict[str, Any]:
        """Process Panopto URL/ID through complete pipeline."""
        # Get meeting name first
        temp_panopto_proc = PanoptoProcessor(Path("."))
        temp_metadata = temp_panopto_proc._extract_metadata(video_id)
        meeting_name = temp_metadata.get('meeting_name', f'Panopto_{video_id[:8]}')
        
        # Create output directory
        output_dir = self._create_output_directory(meeting_name, temp_metadata)
        
        # Process Panopto
        panopto_proc = PanoptoProcessor(output_dir)
        
        if 'video_id' in kwargs:
            kwargs.pop('video_id')
        
        panopto_result = panopto_proc.process(input_path, video_id, meeting_name=meeting_name, **kwargs)
        
        if not panopto_result['success']:
            return panopto_result
        
        output_transcript = Path(panopto_result['transcript_file'])
        video_id = panopto_result['video_id']
        
        # Process through enhanced subtitle processor with speaker recognition
        use_recognition = kwargs.get('speaker_recognition', self.enable_speaker_recognition)
        subtitle_proc = EnhancedSubtitleProcessor(output_dir, enable_speaker_recognition=use_recognition)
        
        # For Panopto, we don't have the original audio, so skip speaker recognition
        subtitle_result = subtitle_proc.process(
            str(output_transcript), 
            audio_path=None,  # No audio available for Panopto
            output_name=meeting_name,
            **kwargs
        )
        
        if not subtitle_result['success']:
            return subtitle_result
        
        # Process transcript
        transcript_proc = TranscriptProcessor(output_dir)
        
        final_result = transcript_proc.process(
            subtitle_result['transcript_file'],
            video_id=video_id,
            metadata={
                **panopto_result['metadata'],
                'meeting_name': meeting_name
            },
            **kwargs
        )
        
        # Note about speaker recognition
        if use_recognition:
            print("\nNote: Speaker recognition requires original audio file.")
            print("Panopto transcripts will use default speaker labels.")
        
        return {
            'success': final_result.get('success', True),
            'input_type': 'panopto',
            'video_id': video_id,
            'meeting_name': meeting_name,
            'output_dir': str(output_dir),
            'files': {
                'original_srt': str(output_transcript),
                'processed_vtt': str(subtitle_result['transcript_file']),
                **final_result.get('files', {})
            },
            'metadata': {
                **panopto_result['metadata'],
                **subtitle_result['metadata'],
                **final_result.get('metadata', {}),
                'speaker_recognition_available': False
            }
        }
    
    def _process_media(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """Process media file through transcription and full pipeline with speaker recognition."""
        media_path = Path(input_path)
        meeting_name = media_path.stem
        
        output_dir = self._create_output_directory(meeting_name, {'source': 'media_file'})
        
        # Use enhanced media processor with speaker recognition
        use_recognition = kwargs.get('speaker_recognition', self.enable_speaker_recognition)
        media_proc = EnhancedMediaProcessor(output_dir, enable_speaker_recognition=use_recognition)
        media_result = media_proc.process(input_path, **kwargs)
        
        if not media_result['success']:
            return media_result
        
        transcript_file = media_result['transcript_file']
        
        # Process through transcript processor
        transcript_proc = TranscriptProcessor(output_dir)
        final_result = transcript_proc.process(
            transcript_file,
            metadata={
                **media_result['metadata'],
                'meeting_name': meeting_name
            },
            **kwargs
        )
        
        # Add speaker summary if available
        result_dict = {
            'success': final_result.get('success', True),
            'input_type': 'media',
            'meeting_name': meeting_name,
            'output_dir': str(output_dir),
            'files': {
                'source_media': input_path,
                'transcript_vtt': str(transcript_file),
                **final_result.get('files', {})
            },
            'metadata': {
                **media_result['metadata'],
                **final_result.get('metadata', {})
            }
        }
        
        # Add speaker information if available
        if 'identified_speakers' in media_result['metadata']:
            result_dict['speakers'] = media_result['metadata']['identified_speakers']
        
        return result_dict
    
    def _process_subtitle(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """Process subtitle file through full pipeline with optional speaker recognition."""
        subtitle_path = Path(input_path)
        meeting_name = subtitle_path.stem
        
        output_dir = self._create_output_directory(meeting_name, {'source': 'subtitle_file'})
        
        # Copy original subtitle
        output_subtitle = output_dir / subtitle_path.name
        shutil.copy2(input_path, str(output_subtitle))
        
        # Use enhanced subtitle processor
        use_recognition = kwargs.get('speaker_recognition', self.enable_speaker_recognition)
        subtitle_proc = EnhancedSubtitleProcessor(output_dir, enable_speaker_recognition=use_recognition)
        
        # Check if audio file is provided for speaker recognition
        audio_path = kwargs.get('audio_path')
        if use_recognition and not audio_path:
            print("\nNote: Speaker recognition requires the original audio file.")
            print("Provide audio_path parameter to enable speaker identification.")
        
        subtitle_result = subtitle_proc.process(
            str(output_subtitle),
            audio_path=audio_path,
            **kwargs
        )
        
        if not subtitle_result['success']:
            return subtitle_result
        
        # Process transcript
        transcript_proc = TranscriptProcessor(output_dir)
        final_result = transcript_proc.process(
            subtitle_result['transcript_file'],
            metadata={
                **subtitle_result['metadata'],
                'meeting_name': meeting_name
            },
            **kwargs
        )
        
        result_dict = {
            'success': final_result.get('success', True),
            'input_type': 'subtitle',
            'meeting_name': meeting_name,
            'output_dir': str(output_dir),
            'files': {
                'source_subtitle': str(output_subtitle),
                'processed_vtt': str(subtitle_result['transcript_file']),
                **final_result.get('files', {})
            },
            'metadata': {
                **subtitle_result['metadata'],
                **final_result.get('metadata', {})
            }
        }
        
        # Add speaker information if available
        if 'identified_speakers' in subtitle_result['metadata']:
            result_dict['speakers'] = subtitle_result['metadata']['identified_speakers']
        
        return result_dict
    
    def _process_excel(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """Process pre-existing Excel file."""
        excel_path = Path(input_path)
        meeting_name = excel_path.stem
        
        output_dir = self._create_output_directory(meeting_name, {'source': 'excel_file'})
        
        output_excel = output_dir / excel_path.name
        shutil.copy2(excel_path, output_excel)
        
        transcript_proc = TranscriptProcessor(output_dir)
        
        video_id = kwargs.get('video_id')
        if not video_id:
            print("Note: No video_id provided. Summaries will be generated without video links.")
        
        final_result = transcript_proc.process_excel(
            output_excel,
            video_id=video_id,
            metadata={'meeting_name': meeting_name},
            **kwargs
        )
        
        return {
            'success': final_result.get('success', True),
            'input_type': 'excel',
            'meeting_name': meeting_name,
            'output_dir': str(output_dir),
            'files': final_result.get('files', {}),
            'metadata': final_result.get('metadata', {})
        }
    
    def _process_directory(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """Process all supported files in a directory."""
        directory = Path(input_path)
        
        if not directory.is_dir():
            return {
                'success': False,
                'error': f'Not a directory: {input_path}'
            }
        
        # Find supported files
        from processors.media_processor_enhanced import EnhancedMediaProcessor
        from processors.subtitle_processor_enhanced import EnhancedSubtitleProcessor
        
        supported_files = []
        media_extensions = {'.mp4', '.mp3', '.wav', '.m4a', '.avi', '.mov', '.flac', '.ogg', '.webm'}
        subtitle_extensions = {'.vtt', '.srt', '.sbv', '.sub'}
        
        for ext in [*media_extensions, *subtitle_extensions, '.xlsx']:
            supported_files.extend(directory.glob(f'*{ext}'))
        
        if not supported_files:
            return {
                'success': False,
                'error': f'No supported files found in directory: {input_path}'
            }
        
        # Process each file
        results = []
        for file_path in supported_files:
            print(f"\nProcessing: {file_path.name}")
            result = self.process(str(file_path), **kwargs)
            results.append({
                'file': str(file_path),
                'result': result
            })
        
        successful = sum(1 for r in results if r['result'].get('success', False))
        
        return {
            'success': successful > 0,
            'input_type': 'directory',
            'total_files': len(supported_files),
            'successful': successful,
            'failed': len(supported_files) - successful,
            'results': results
        }