# File: processors/subtitle_processor_enhanced.py
"""
Enhanced Subtitle Processor with integrated speaker recognition.
Extends the base subtitle processor to identify speakers using voice fingerprinting.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from processors.subtitle_processor import SubtitleProcessor
from speaker_id.fingerprint_manager import SpeakerFingerprintManager
from core.config import Config


class EnhancedSubtitleProcessor(SubtitleProcessor):
    """Subtitle processor with speaker recognition capabilities."""
    
    def __init__(self, output_dir: Path, enable_speaker_recognition: bool = True):
        """
        Initialize enhanced subtitle processor.
        
        Args:
            output_dir: Output directory for processed files
            enable_speaker_recognition: Enable speaker fingerprinting
        """
        super().__init__(output_dir)
        
        self.enable_speaker_recognition = enable_speaker_recognition
        self.speaker_manager = None
        
        if enable_speaker_recognition:
            try:
                self.speaker_manager = SpeakerFingerprintManager()
                print("Speaker recognition enabled")
            except Exception as e:
                print(f"Warning: Could not initialize speaker recognition: {e}")
                self.enable_speaker_recognition = False
    
    def process(self, input_path: str, audio_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process subtitle file with optional speaker recognition.
        
        Args:
            input_path: Path to subtitle file
            audio_path: Path to original audio file (required for speaker recognition)
            **kwargs: Additional options
            
        Returns:
            Dict with processing results including identified speakers
        """
        # First, process normally
        result = super().process(input_path, **kwargs)
        
        if not result['success']:
            return result
        
        # Apply speaker recognition if enabled and audio provided
        if self.enable_speaker_recognition and self.speaker_manager and audio_path:
            segments = result.get('segments', [])
            
            if segments:
                print(f"Applying speaker recognition to {len(segments)} segments...")
                
                # Process segments for speaker identification
                updated_segments = self.speaker_manager.process_diarized_segments(
                    audio_path, 
                    segments,
                    auto_add_unknown=kwargs.get('auto_add_speakers', True)
                )
                
                # Update the segments in result
                result['segments'] = updated_segments
                
                # Re-save the VTT file with identified speakers
                output_name = kwargs.get('output_name', Path(input_path).stem)
                output_vtt = self.output_dir / f"{output_name}_processed.vtt"
                self._save_as_vtt(updated_segments, output_vtt)
                
                # Update metadata with speaker information
                identified_speakers = self._extract_speaker_info(updated_segments)
                result['metadata']['identified_speakers'] = identified_speakers
                result['metadata']['speaker_recognition_applied'] = True
                
                # Save updated metadata
                self.save_metadata(result['metadata'])
                
                print(f"Speaker recognition complete. Identified {len(identified_speakers)} unique speakers.")
        
        return result
    
    def _extract_speaker_info(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Extract information about identified speakers."""
        speakers = {}
        
        for segment in segments:
            speaker_name = segment.get('speaker', 'Unknown')
            speaker_id = segment.get('speaker_id')
            confidence = segment.get('speaker_confidence', 0.0)
            is_new = segment.get('speaker_is_new', False)
            
            if speaker_name not in speakers:
                speakers[speaker_name] = {
                    'name': speaker_name,
                    'id': speaker_id,
                    'confidence_scores': [],
                    'segment_count': 0,
                    'total_duration': 0.0,
                    'is_new': is_new
                }
            
            speakers[speaker_name]['segment_count'] += 1
            speakers[speaker_name]['confidence_scores'].append(confidence)
            speakers[speaker_name]['total_duration'] += (
                segment.get('end', 0) - segment.get('start', 0)
            )
        
        # Calculate average confidence for each speaker
        speaker_list = []
        for speaker_name, info in speakers.items():
            avg_confidence = sum(info['confidence_scores']) / len(info['confidence_scores'])
            speaker_list.append({
                'name': speaker_name,
                'id': info['id'],
                'average_confidence': avg_confidence,
                'segment_count': info['segment_count'],
                'total_duration': round(info['total_duration'], 2),
                'is_new': info['is_new']
            })
        
        # Sort by total duration (most speaking time first)
        speaker_list.sort(key=lambda x: x['total_duration'], reverse=True)
        
        return speaker_list