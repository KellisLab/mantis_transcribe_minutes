#!/usr/bin/env python3
"""
Speaker Auto-Correction Script
Automatically corrects generic speaker labels by matching with a reference transcript
Updated to work with multi-sample speaker database and interactive prompting
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import json


class TranscriptSegment:
    """Represents a single transcript segment."""
    def __init__(self, timestamp: str, speaker: str, text: str):
        self.timestamp = timestamp
        self.speaker = speaker
        self.text = text
        self.seconds = self._time_to_seconds(timestamp)
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convert HH:MM:SS to seconds."""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    
    def __repr__(self):
        return f"<Segment {self.timestamp} {self.speaker}: {self.text[:30]}...>"


class SpeakerCorrector:
    """Main class for correcting speaker labels."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.speaker_mapping = {}
        self.confidence_scores = {}
    
    def parse_transcript(self, content: str) -> List[TranscriptSegment]:
        """Parse transcript content into segments."""
        segments = []
        
        # Pattern: HH:MM:SS Speaker_Name: Text
        pattern = r'(\d{2}:\d{2}:\d{2})\s+([^:]+):\s*(.+)'
        
        for line in content.strip().split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                timestamp, speaker, text = match.groups()
                segments.append(TranscriptSegment(
                    timestamp.strip(),
                    speaker.strip(),
                    text.strip()
                ))
        
        return segments
    
    def find_best_match(self, target_segment: TranscriptSegment, 
                       reference_segments: List[TranscriptSegment],
                       time_window: int = 5) -> Optional[TranscriptSegment]:
        """Find the best matching segment in reference transcript."""
        best_match = None
        best_score = 0
        
        # Consider segments within time window
        candidates = [
            seg for seg in reference_segments
            if abs(seg.seconds - target_segment.seconds) <= time_window
        ]
        
        for candidate in candidates:
            # Calculate text similarity
            score = SequenceMatcher(None, 
                                  target_segment.text.lower(), 
                                  candidate.text.lower()).ratio()
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = candidate
        
        return best_match
    
    def build_speaker_mapping(self, generic_segments: List[TranscriptSegment],
                            reference_segments: List[TranscriptSegment]) -> Dict[str, str]:
        """Build mapping from generic speaker labels to actual names."""
        # Track all matches
        matches = defaultdict(list)
        
        for gen_segment in generic_segments:
            ref_match = self.find_best_match(gen_segment, reference_segments)
            if ref_match:
                matches[gen_segment.speaker].append(ref_match.speaker)
        
        # Determine most common mapping for each generic speaker
        speaker_mapping = {}
        confidence_scores = {}
        
        for generic_speaker, matched_names in matches.items():
            if matched_names:
                # Count occurrences
                name_counts = Counter(matched_names)
                most_common = name_counts.most_common(1)[0]
                
                speaker_mapping[generic_speaker] = most_common[0]
                confidence_scores[generic_speaker] = most_common[1] / len(matched_names)
        
        self.speaker_mapping = speaker_mapping
        self.confidence_scores = confidence_scores
        
        return speaker_mapping
    
    def apply_corrections(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Apply speaker mapping to correct segments."""
        corrected = []
        
        for segment in segments:
            new_speaker = self.speaker_mapping.get(segment.speaker, segment.speaker)
            corrected.append(TranscriptSegment(
                segment.timestamp,
                new_speaker,
                segment.text
            ))
        
        return corrected
    
    def generate_report(self) -> str:
        """Generate a report of the corrections made."""
        report = ["SPEAKER CORRECTION REPORT", "=" * 50, ""]
        
        if self.speaker_mapping:
            report.append("Speaker Mappings Found:")
            for generic, actual in sorted(self.speaker_mapping.items()):
                confidence = self.confidence_scores.get(generic, 0)
                report.append(f"  {generic} -> {actual} (confidence: {confidence:.1%})")
        else:
            report.append("No speaker mappings found!")
            report.append("This might happen if:")
            report.append("  - The transcripts don't overlap in time")
            report.append("  - The text content is too different")
            report.append("  - The similarity threshold is too high")
        
        return "\n".join(report)
    
    def correct_transcript_file(self, generic_file: Path, reference_file: Path,
                              output_file: Optional[Path] = None) -> Path:
        """Main method to correct a transcript file."""
        # Read files
        with open(generic_file, 'r', encoding='utf-8') as f:
            generic_content = f.read()
        
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference_content = f.read()
        
        # Parse transcripts
        print("Parsing transcripts...")
        generic_segments = self.parse_transcript(generic_content)
        reference_segments = self.parse_transcript(reference_content)
        
        print(f"Found {len(generic_segments)} segments in generic transcript")
        print(f"Found {len(reference_segments)} segments in reference transcript")
        
        # Build speaker mapping
        print("\nBuilding speaker mapping...")
        mapping = self.build_speaker_mapping(generic_segments, reference_segments)
        
        # Apply corrections
        print("Applying corrections...")
        corrected_segments = self.apply_corrections(generic_segments)
        
        # Generate output
        if output_file is None:
            output_file = generic_file.parent / f"{generic_file.stem}_corrected.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in corrected_segments:
                f.write(f"{segment.timestamp} {segment.speaker}: {segment.text}\n")
        
        # Print report
        print("\n" + self.generate_report())
        
        # Save mapping for future use
        mapping_file = output_file.parent / f"{output_file.stem}_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump({
                'mapping': self.speaker_mapping,
                'confidence': self.confidence_scores
            }, f, indent=2)
        
        print(f"\nCorrected transcript saved to: {output_file}")
        print(f"Speaker mapping saved to: {mapping_file}")
        
        return output_file


def merge_speaker_databases(corrector: SpeakerCorrector, speaker_db_path: Optional[Path] = None,
                           max_samples: int = 5) -> bool:
    """
    Merge speakers in the database based on the mapping.
    Updated to work with multi-sample speaker database.
    
    Returns: True if merge was successful, False otherwise
    """
    try:
        from speaker_id.fingerprint_manager import SpeakerFingerprintManager
        from tabulate import tabulate
        
        manager = SpeakerFingerprintManager(
            db_path=speaker_db_path,
            max_samples_per_speaker=max_samples
        )
        
        print("\n" + "="*60)
        print("SPEAKER DATABASE UPDATE")
        print("="*60)
        
        # Find speakers in database
        all_speakers = manager.list_speakers()
        
        # Analyze what changes would be made
        changes_to_make = []
        
        for generic_name, actual_name in corrector.speaker_mapping.items():
            generic_speakers = [s for s in all_speakers if s['name'] == generic_name]
            actual_speakers = [s for s in all_speakers if s['name'] == actual_name]
            
            if generic_speakers and not actual_speakers:
                # Would rename generic to actual
                for speaker in generic_speakers:
                    changes_to_make.append({
                        'action': 'RENAME',
                        'from': generic_name,
                        'to': actual_name,
                        'samples': speaker['sample_count'],
                        'confidence': f"{corrector.confidence_scores.get(generic_name, 0):.0%}"
                    })
            
            elif generic_speakers and actual_speakers:
                # Would merge generic into actual
                total_samples = sum(s['sample_count'] for s in generic_speakers)
                existing_samples = sum(s['sample_count'] for s in actual_speakers)
                changes_to_make.append({
                    'action': 'MERGE',
                    'from': f"{generic_name} ({total_samples} samples)",
                    'to': f"{actual_name} ({existing_samples} samples)",
                    'samples': f"{existing_samples + total_samples} total",
                    'confidence': f"{corrector.confidence_scores.get(generic_name, 0):.0%}"
                })
        
        if not changes_to_make:
            print("\nNo changes needed - all speakers already correctly named!")
            return True
        
        # Display proposed changes
        print("\nProposed Database Changes:")
        print(tabulate(changes_to_make, headers="keys", tablefmt="grid"))
        
        # Show current database stats
        print(f"\nCurrent Database Stats:")
        print(f"  Total speakers: {len(all_speakers)}")
        print(f"  Total samples: {sum(s['sample_count'] for s in all_speakers)}")
        print(f"  Max samples per speaker: {manager.max_samples_per_speaker}")
        
        # Ask for confirmation
        print("\n" + "="*60)
        response = input("Do you want to apply these changes to the speaker database? (y/N): ")
        
        if response.lower() != 'y':
            print("Database update cancelled.")
            return False
        
        # Apply changes
        print("\nApplying database changes...")
        successful_changes = 0
        
        for generic_name, actual_name in corrector.speaker_mapping.items():
            generic_speakers = [s for s in all_speakers if s['name'] == generic_name]
            actual_speakers = [s for s in all_speakers if s['name'] == actual_name]
            
            try:
                if generic_speakers and not actual_speakers:
                    # Rename generic to actual
                    for speaker in generic_speakers:
                        print(f"  Renaming {generic_name} to {actual_name}...")
                        if manager.rename_speaker(speaker['id'], actual_name):
                            successful_changes += 1
                
                elif generic_speakers and actual_speakers:
                    # Merge generic into actual
                    speaker_ids = [s['id'] for s in generic_speakers] + [s['id'] for s in actual_speakers]
                    if len(speaker_ids) > 1:
                        print(f"  Merging {generic_name} into {actual_name}...")
                        result = manager.merge_speakers(speaker_ids, actual_name)
                        if result:
                            successful_changes += 1
            
            except Exception as e:
                print(f"  Error processing {generic_name}: {e}")
        
        # Show results
        print(f"\nDatabase update complete!")
        print(f"  Changes applied: {successful_changes}")
        
        # Show updated stats
        updated_speakers = manager.list_speakers()
        print(f"\nUpdated Database Stats:")
        print(f"  Total speakers: {len(updated_speakers)}")
        print(f"  Total samples: {sum(s['sample_count'] for s in updated_speakers)}")
        
        # Show speakers with most samples
        top_speakers = sorted(updated_speakers, key=lambda x: x['sample_count'], reverse=True)[:5]
        if top_speakers:
            print("\nTop speakers by sample count:")
            for speaker in top_speakers:
                quality = "★" * min(speaker['sample_count'], 5)
                print(f"  - {speaker['name']}: {speaker['sample_count']} samples {quality}")
        
        return True
        
    except ImportError:
        print("\n" + "="*60)
        print("Speaker database module not available.")
        print("To update the speaker database, install the required dependencies:")
        print("  pip install speechbrain chromadb")
        print("="*60)
        return False
    except Exception as e:
        print(f"\nError accessing speaker database: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automatically correct speaker labels by matching with reference transcript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic correction
  python speaker_correct.py generic.txt reference.txt
  
  # With custom output
  python speaker_correct.py generic.txt reference.txt -o corrected.txt
  
  # With custom similarity threshold
  python speaker_correct.py generic.txt reference.txt --threshold 0.9
  
  # Skip database prompt
  python speaker_correct.py generic.txt reference.txt --no-db-prompt
        """
    )
    
    parser.add_argument('generic_file', type=Path,
                       help='Transcript file with generic speaker labels')
    parser.add_argument('reference_file', type=Path,
                       help='Reference transcript with actual speaker names')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output file path (default: generic_corrected.txt)')
    parser.add_argument('--threshold', type=float, default=0.85,
                       help='Similarity threshold for matching (0-1, default: 0.85)')
    parser.add_argument('--db-path', type=Path,
                       help='Custom speaker database path')
    parser.add_argument('--time-window', type=int, default=5,
                       help='Time window in seconds for matching (default: 5)')
    parser.add_argument('--no-db-prompt', action='store_true',
                       help='Skip speaker database update prompt')
    parser.add_argument('--max-samples', type=int, default=5,
                       help='Max samples per speaker in database (default: 5)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.generic_file.exists():
        print(f"Error: Generic file not found: {args.generic_file}")
        return 1
    
    if not args.reference_file.exists():
        print(f"Error: Reference file not found: {args.reference_file}")
        return 1
    
    # Create corrector
    corrector = SpeakerCorrector(similarity_threshold=args.threshold)
    
    try:
        # Correct transcript
        output_file = corrector.correct_transcript_file(
            args.generic_file,
            args.reference_file,
            args.output
        )
        
        # Check if we found any mappings
        if corrector.speaker_mapping and not args.no_db_prompt:
            # Prompt for database update
            print("\n" + "="*60)
            print("SPEAKER DATABASE UPDATE OPTION")
            print("="*60)
            print("The speaker correction found mappings that could be applied")
            print("to your speaker recognition database.")
            print("")
            response = input("Would you like to check the speaker database for updates? (y/N): ")
            
            if response.lower() == 'y':
                merge_speaker_databases(corrector, args.db_path, args.max_samples)
        elif not corrector.speaker_mapping:
            print("\n" + "="*60)
            print("No speaker mappings were found.")
            print("The speaker database will not be modified.")
            print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())