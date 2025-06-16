# File: cli_enhanced.py
#!/usr/bin/env python3
"""
Unified Transcript Processing Platform CLI - With Speaker Recognition
"""
import argparse
import sys
import os
from pathlib import Path

from core.config import Config
from core.input_detector import InputDetector, InputType
from processors.pipeline_processor import PipelineProcessor

def main():
    parser = argparse.ArgumentParser(
        description="Process transcripts from various sources with speaker recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with speaker recognition (default)
  python cli.py video.mp4
  
  # Process without speaker recognition
  python cli.py video.mp4 --no-speaker-recognition
  
  # Process subtitle with audio for speaker recognition
  python cli.py transcript.vtt --audio-path original_audio.mp4
  
  # Auto-add unknown speakers (default behavior)
  python cli.py video.mp4 --auto-add-speakers
  
  # Process Panopto URL (no speaker recognition available)
  python cli.py "https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=VIDEO_ID"
  
  # Process directory of files
  python cli.py ./recordings/ --speaker-recognition
  
Speaker Management:
  # List speakers in database
  python speaker_cli.py list
  
  # Rename a speaker
  python speaker_cli.py rename "Speaker_1" "John Doe"
  
  # Merge duplicate speakers
  python speaker_cli.py merge Speaker_1 Speaker_2 --target "John Doe"
        """
    )
    
    parser.add_argument(
        "input",
        help="Input: Panopto URL/ID, media file, subtitle file, or directory"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: from .env or ./output)"
    )
    parser.add_argument(
        "--video-id",
        help="Panopto video ID for generating clickable links (for local files)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for transcription (default: en)"
    )
    
    # Speaker recognition options
    speaker_group = parser.add_argument_group("Speaker Recognition")
    speaker_group.add_argument(
        "--no-speaker-recognition",
        action="store_true",
        help="Disable speaker recognition (faster processing)"
    )
    speaker_group.add_argument(
        "--speaker-recognition",
        action="store_true",
        help="Force enable speaker recognition"
    )
    speaker_group.add_argument(
        "--auto-add-speakers",
        action="store_true",
        default=True,
        help="Automatically add unknown speakers to database (default)"
    )
    speaker_group.add_argument(
        "--no-auto-add-speakers",
        action="store_false",
        dest="auto_add_speakers",
        help="Don't automatically add unknown speakers"
    )
    speaker_group.add_argument(
        "--audio-path",
        help="Path to audio file (required for speaker recognition with subtitle files)"
    )
    speaker_group.add_argument(
        "--speaker-db-path",
        type=Path,
        help="Custom path for speaker database"
    )
    
    # Processing options
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Skip timestamp refinement step"
    )
    parser.add_argument(
        "--skip-diarization",
        action="store_true", 
        help="Skip speaker diarization (disables speaker recognition)"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect input type without processing"
    )
    parser.add_argument(
        "--whisper-model",
        help="Override WhisperX model size from .env"
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    Config.setup_directories()
    
    # Override config if needed
    if args.whisper_model:
        Config.WHISPER_MODEL = args.whisper_model
    
    # Detect input type
    input_type, metadata = InputDetector.detect(args.input)
    
    print(f"Detected input type: {input_type.value}")
    if metadata:
        print(f"Metadata: {metadata}")
    
    if args.detect_only:
        return 0
    
    if input_type == InputType.UNKNOWN:
        print(f"Error: Unable to determine input type for '{args.input}'")
        return 1
    
    # Determine speaker recognition setting
    enable_speaker_recognition = True  # Default
    if args.no_speaker_recognition:
        enable_speaker_recognition = False
    elif args.speaker_recognition:
        enable_speaker_recognition = True
    elif args.skip_diarization:
        enable_speaker_recognition = False
        print("Note: Speaker recognition disabled due to --skip-diarization")
    
    # Check for special cases
    if input_type in [InputType.PANOPTO_URL, InputType.PANOPTO_ID]:
        if enable_speaker_recognition:
            print("\nNote: Speaker recognition requires original audio.")
            print("Panopto transcripts will use default speaker labels.")
    
    if input_type == InputType.SUBTITLE_FILE and enable_speaker_recognition and not args.audio_path:
        print("\nWarning: Speaker recognition requires original audio file.")
        print("Use --audio-path to provide the audio for speaker identification.")
        enable_speaker_recognition = False
    
    # Process through pipeline
    try:
        pipeline = PipelineProcessor(
            Path(args.output) if args.output else None,
            enable_speaker_recognition=enable_speaker_recognition
        )
        
        # Prepare kwargs
        process_kwargs = {
            'language': args.language,
            'skip_refinement': args.skip_refinement,
            'skip_diarization': args.skip_diarization,
            'speaker_recognition': enable_speaker_recognition,
            'auto_add_speakers': args.auto_add_speakers,
        }
        
        # Add audio path for subtitle processing
        if args.audio_path:
            process_kwargs['audio_path'] = args.audio_path
        
        # Add speaker database path if specified
        if args.speaker_db_path:
            process_kwargs['speaker_db_path'] = args.speaker_db_path
        
        # Only add video_id for non-Panopto inputs
        if input_type not in [InputType.PANOPTO_URL, InputType.PANOPTO_ID]:
            if args.video_id:
                process_kwargs['video_id'] = args.video_id
        
        # Process
        result = pipeline.process(args.input, **process_kwargs)
        
        if result['success']:
            print("\n✓ Processing completed successfully!")
            print(f"Output directory: {result.get('output_dir', 'N/A')}")
            
            # Display generated files
            if 'files' in result:
                print("\nGenerated files:")
                for file_type, file_path in result['files'].items():
                    print(f"  - {file_type}: {file_path}")
            
            # Display speaker information if available
            if 'speakers' in result:
                print(f"\nIdentified {len(result['speakers'])} speakers:")
                for speaker in result['speakers'][:5]:  # Show top 5
                    status = "NEW" if speaker.get('is_new') else "KNOWN"
                    confidence = speaker.get('average_confidence', 0)
                    print(f"  - {speaker['name']} [{status}]: "
                          f"{speaker['total_duration']}s ({confidence:.0%} confidence)")
                
                if len(result['speakers']) > 5:
                    print(f"  ... and {len(result['speakers']) - 5} more")
                
                print("\nUse 'python speaker_cli.py list' to manage speakers")
        else:
            print(f"\n✗ Processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())