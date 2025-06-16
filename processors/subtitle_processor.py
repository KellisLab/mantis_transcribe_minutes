# File: processors/subtitle_processor.py
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from processors.base_processor import BaseProcessor

class SubtitleProcessor(BaseProcessor):
    """Processor for parsing VTT/SRT subtitle files."""
    
    def process(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process subtitle file to extract structured transcript data.
        
        Args:
            input_path: Path to subtitle file
            **kwargs: Additional options (output_name, etc.)
        
        Returns:
            Dict with processing results
        """
        input_file = Path(input_path)
        
        if not input_file.exists():
            return {
                'success': False,
                'error': f'Subtitle file not found: {input_path}'
            }
        
        # Detect format
        file_ext = input_file.suffix.lower()
        
        try:
            if file_ext == '.vtt':
                segments = self._parse_vtt(input_file)
            elif file_ext == '.srt':
                segments = self._parse_srt(input_file)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported subtitle format: {file_ext}'
                }
            
            # Extract speakers if present
            speakers = self._extract_speakers(segments)
            
            # Save as standardized VTT with proper name
            output_name = kwargs.get('output_name', input_file.stem)
            output_vtt = self.output_dir / f"{output_name}_processed.vtt"
            self._save_as_vtt(segments, output_vtt)
            
            # Save metadata
            metadata = {
                'source_file': str(input_file),
                'format': file_ext,
                'segments_count': len(segments),
                'speakers': list(speakers),
                'has_speakers': len(speakers) > 0,
                'transcript_file': str(output_vtt)
            }
            self.save_metadata(metadata)
            
            return {
                'success': True,
                'transcript_file': output_vtt,
                'metadata': metadata,
                'segments': segments
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to parse subtitle file: {str(e)}'
            }
    
    def _parse_vtt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse WebVTT file."""
        segments = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove WEBVTT header
        content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
        
        # Split into cues
        cues = re.split(r'\n\n+', content.strip())
        
        for cue in cues:
            if not cue.strip():
                continue
            
            lines = cue.strip().split('\n')
            if len(lines) < 2:
                continue
            
            # Parse timestamp line
            timestamp_line = lines[0] if '-->' in lines[0] else lines[1] if len(lines) > 1 and '-->' in lines[1] else None
            
            if not timestamp_line:
                continue
            
            # Parse timestamps
            match = re.match(r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})', timestamp_line)
            if not match:
                continue
            
            start_time = self._vtt_time_to_seconds(match.group(1))
            end_time = self._vtt_time_to_seconds(match.group(2))
            
            # Get text (skip cue ID if present)
            text_start = 1 if '-->' in lines[0] else 2
            text_lines = lines[text_start:]
            
            # Skip Panopto disclaimer line if present
            if text_lines and "[Auto-generated transcript. Edits may have been applied for clarity.]" in text_lines[0]:
                text_lines = text_lines[1:]
            
            text = ' '.join(text_lines)
            
            # Extract speaker if present (format: "Speaker: text" or "SPEAKER_00: text")
            speaker = "Unknown"
            speaker_match = re.match(r'^([^:]+?):\s*(.+)', text)
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                text = speaker_match.group(2).strip()
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'speaker': speaker,
                'text': text
            })
        
        return segments
    
    def _parse_srt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse SRT file."""
        segments = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into subtitle blocks
        blocks = re.split(r'\n\n+', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # Parse timestamp line (should be second line)
            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
            if not timestamp_match:
                continue
            
            start_time = self._srt_time_to_seconds(timestamp_match.group(1))
            end_time = self._srt_time_to_seconds(timestamp_match.group(2))
            
            # Get text (remaining lines)
            text_lines = lines[2:]
            
            # Skip Panopto disclaimer line if it's the first text line
            if text_lines and "[Auto-generated transcript. Edits may have been applied for clarity.]" in text_lines[0]:
                text_lines = text_lines[1:]
            
            text = ' '.join(text_lines)
            
            # Extract speaker if present
            speaker = "Unknown"
            speaker_match = re.match(r'^([^:]+?):\s*(.+)', text)
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                text = speaker_match.group(2).strip()
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'speaker': speaker,
                'text': text
            })
        
        return segments
    
    def _vtt_time_to_seconds(self, time_str: str) -> float:
        """Convert VTT timestamp to seconds."""
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    
    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT timestamp to seconds."""
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    
    def _extract_speakers(self, segments: List[Dict]) -> set:
        """Extract unique speakers from segments."""
        return {seg['speaker'] for seg in segments if seg['speaker'] != 'Unknown'}
    
    def _save_as_vtt(self, segments: List[Dict], output_path: Path):
        """Save segments as VTT file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for i, seg in enumerate(segments, 1):
                # Convert times back to VTT format
                start = self._seconds_to_vtt_time(seg['start'])
                end = self._seconds_to_vtt_time(seg['end'])
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                
                # Include speaker in text if not Unknown
                if seg['speaker'] != 'Unknown':
                    f.write(f"{seg['speaker']}: {seg['text']}\n\n")
                else:
                    f.write(f"{seg['text']}\n\n")
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to VTT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"