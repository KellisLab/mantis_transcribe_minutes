# File: processors/media_processor_enhanced.py
"""
Enhanced Media Processor with integrated speaker recognition.
Extends WhisperX processing to identify speakers by voice.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from processors.media_processor import MediaProcessor, WHISPERX_AVAILABLE
from speaker_id.fingerprint_manager import SpeakerFingerprintManager
from core.config import Config

# Import WhisperX components if available
if WHISPERX_AVAILABLE:
    import whisperx
    import torch


class EnhancedMediaProcessor(MediaProcessor):
    """Media processor with speaker recognition capabilities."""
    
    def __init__(self, output_dir: Path, enable_speaker_recognition: bool = True):
        """
        Initialize enhanced media processor.
        
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
                print("Speaker recognition enabled for media processing")
            except Exception as e:
                print(f"Warning: Could not initialize speaker recognition: {e}")
                self.enable_speaker_recognition = False
    
    def process(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process media file with WhisperX and speaker recognition.
        
        Args:
            input_path: Path to media file
            **kwargs: Additional options including:
                - language: Language code
                - skip_diarization: Skip diarization step
                - auto_add_speakers: Auto-add unknown speakers
                - speaker_recognition: Override enable_speaker_recognition
            
        Returns:
            Dict with processing results including identified speakers
        """
        # Check if speaker recognition should be applied
        apply_recognition = kwargs.get('speaker_recognition', self.enable_speaker_recognition)
        
        # Skip diarization if no HF token and recognition is requested
        if not Config.HF_TOKEN and apply_recognition:
            print("Warning: Speaker recognition requires diarization. Setting up diarization...")
            kwargs['skip_diarization'] = False
        
        # Process with base media processor
        result = super().process(input_path, **kwargs)
        
        if not result['success']:
            return result
        
        # Apply speaker recognition if enabled
        if apply_recognition and self.speaker_manager and not kwargs.get('skip_diarization', False):
            segments = result.get('segments', [])
            
            if segments and self._has_speaker_info(segments):
                print(f"\nApplying speaker recognition to {len(segments)} segments...")
                
                # Convert WhisperX segments to our format
                converted_segments = self._convert_whisperx_segments(segments)
                
                # Process segments for speaker identification
                updated_segments = self.speaker_manager.process_diarized_segments(
                    input_path,
                    converted_segments,
                    auto_add_unknown=kwargs.get('auto_add_speakers', True)
                )
                
                # Map back to original segments
                result['segments'] = self._update_original_segments(segments, updated_segments)
                
                # Re-save the transcript with identified speakers
                srt_file = self._save_enhanced_transcript(result, Path(input_path).stem)
                result['transcript_file'] = srt_file
                
                # Update metadata
                identified_speakers = self._extract_speaker_info(updated_segments)
                result['metadata']['identified_speakers'] = identified_speakers
                result['metadata']['speaker_recognition_applied'] = True
                
                # Save updated metadata
                self.save_metadata(result['metadata'])
                
                print(f"Speaker recognition complete. Identified {len(identified_speakers)} unique speakers.")
                
                # Print speaker summary
                self._print_speaker_summary(identified_speakers)
        
        return result
    
    def _has_speaker_info(self, segments: List[Dict]) -> bool:
        """Check if segments contain speaker information."""
        return any('speaker' in seg for seg in segments)
    
    def _convert_whisperx_segments(self, whisperx_segments: List[Dict]) -> List[Dict]:
        """Convert WhisperX segments to our standard format."""
        converted = []
        
        for seg in whisperx_segments:
            # WhisperX format might have different keys
            converted_seg = {
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': seg.get('text', ''),
                'speaker': seg.get('speaker', 'SPEAKER_00')  # Default if no speaker
            }
            
            # Copy any additional fields
            for key in ['words', 'avg_logprob', 'compression_ratio', 'no_speech_prob']:
                if key in seg:
                    converted_seg[key] = seg[key]
            
            converted.append(converted_seg)
        
        return converted
    
    def _update_original_segments(self, original_segments: List[Dict], 
                                 updated_segments: List[Dict]) -> List[Dict]:
        """Update original WhisperX segments with speaker identification results."""
        # Create a mapping based on timestamps
        update_map = {}
        for seg in updated_segments:
            key = (seg['start'], seg['end'])
            update_map[key] = seg
        
        # Update original segments
        for i, seg in enumerate(original_segments):
            key = (seg.get('start', 0), seg.get('end', 0))
            if key in update_map:
                updated = update_map[key]
                seg['speaker'] = updated.get('speaker', seg.get('speaker', 'Unknown'))
                seg['speaker_id'] = updated.get('speaker_id')
                seg['speaker_confidence'] = updated.get('speaker_confidence', 0.0)
                seg['speaker_is_new'] = updated.get('speaker_is_new', False)
        
        return original_segments
    
    def _save_enhanced_transcript(self, result: Dict, base_name: str) -> Path:
        """Save transcript with enhanced speaker information."""
        from whisperx.utils import get_writer
        
        # Prepare enhanced result with speaker names
        enhanced_result = result.copy()
        
        # Ensure language key exists
        enhanced_result["language"] = enhanced_result.get("language", "en")
        
        # Write SRT file with enhanced speaker info
        output_format = "srt"
        options = {
            "highlight_words": False,
            "max_line_width": None,
            "max_line_count": None
        }
        
        # Create writer
        writer = get_writer(output_format, str(self.output_dir))
        
        # Create a mock audio path for the writer
        mock_audio_path = str(self.output_dir / f"{base_name}_enhanced.tmp")
        writer(enhanced_result, mock_audio_path, options)
        
        # The writer creates a file with the audio file's base name
        srt_file = self.output_dir / f"{base_name}_enhanced.srt"
        
        # Also create a detailed transcript with confidence scores
        self._save_detailed_transcript(result['segments'], base_name)
        
        print(f"Enhanced transcript saved to: {srt_file}")
        return srt_file
    
    def _save_detailed_transcript(self, segments: List[Dict], base_name: str) -> Path:
        """Save detailed transcript with speaker confidence scores."""
        detail_file = self.output_dir / f"{base_name}_speaker_details.txt"
        
        with open(detail_file, 'w', encoding='utf-8') as f:
            f.write("SPEAKER IDENTIFICATION DETAILS\n")
            f.write("=" * 50 + "\n\n")
            
            current_speaker = None
            for seg in segments:
                speaker = seg.get('speaker', 'Unknown')
                confidence = seg.get('speaker_confidence', 0.0)
                is_new = seg.get('speaker_is_new', False)
                
                # Write speaker header if changed
                if speaker != current_speaker:
                    current_speaker = speaker
                    f.write(f"\n[{speaker}]")
                    if confidence > 0:
                        f.write(f" (confidence: {confidence:.2%})")
                    if is_new:
                        f.write(" *NEW SPEAKER*")
                    f.write("\n")
                
                # Write segment
                start_time = self._format_time(seg.get('start', 0))
                end_time = self._format_time(seg.get('end', 0))
                text = seg.get('text', '').strip()
                
                f.write(f"{start_time} --> {end_time}: {text}\n")
        
        return detail_file
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
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
                    'is_new': is_new,
                    'first_appearance': segment.get('start', 0)
                }
            
            speakers[speaker_name]['segment_count'] += 1
            speakers[speaker_name]['confidence_scores'].append(confidence)
            speakers[speaker_name]['total_duration'] += (
                segment.get('end', 0) - segment.get('start', 0)
            )
        
        # Calculate statistics for each speaker
        speaker_list = []
        for speaker_name, info in speakers.items():
            avg_confidence = sum(info['confidence_scores']) / len(info['confidence_scores'])
            speaker_list.append({
                'name': speaker_name,
                'id': info['id'],
                'average_confidence': avg_confidence,
                'segment_count': info['segment_count'],
                'total_duration': round(info['total_duration'], 2),
                'speaking_time_percentage': 0.0,  # Will calculate below
                'is_new': info['is_new'],
                'first_appearance': info['first_appearance']
            })
        
        # Calculate speaking time percentages
        total_duration = sum(s['total_duration'] for s in speaker_list)
        if total_duration > 0:
            for speaker in speaker_list:
                speaker['speaking_time_percentage'] = round(
                    (speaker['total_duration'] / total_duration) * 100, 1
                )
        
        # Sort by total duration (most speaking time first)
        speaker_list.sort(key=lambda x: x['total_duration'], reverse=True)
        
        return speaker_list
    
    def _print_speaker_summary(self, speakers: List[Dict[str, Any]]):
        """Print a summary of identified speakers."""
        print("\nSPEAKER SUMMARY")
        print("=" * 60)
        
        for i, speaker in enumerate(speakers, 1):
            status = "NEW" if speaker['is_new'] else "KNOWN"
            print(f"\n{i}. {speaker['name']} [{status}]")
            print(f"   - Speaking time: {speaker['total_duration']:.1f}s ({speaker['speaking_time_percentage']:.1f}%)")
            print(f"   - Segments: {speaker['segment_count']}")
            print(f"   - Confidence: {speaker['average_confidence']:.1%}")
            print(f"   - First appears at: {self._format_time(speaker['first_appearance'])}")
        
        print("\n" + "=" * 60)