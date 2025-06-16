# File: speaker_cli.py
#!/usr/bin/env python3
"""
Speaker Management CLI - Manage speaker database for voice recognition
"""

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

from speaker_id.fingerprint_manager import SpeakerFingerprintManager
from core.config import Config


def list_speakers(args):
    """List all speakers in the database."""
    manager = SpeakerFingerprintManager(db_path=args.db_path)
    speakers = manager.list_speakers()
    
    if not speakers:
        print("No speakers in database.")
        return
    
    # Prepare table data
    headers = ["#", "Name", "ID", "First Seen", "Last Seen", "Samples"]
    rows = []
    
    for i, speaker in enumerate(speakers, 1):
        rows.append([
            i,
            speaker['name'],
            speaker['id'][:8] + "...",  # Truncate ID for display
            speaker['first_seen'][:10] if speaker['first_seen'] else "N/A",
            speaker['last_seen'][:10] if speaker['last_seen'] else "N/A",
            speaker['sample_count']
        ])
    
    print(f"\nTotal speakers: {len(speakers)}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def rename_speaker(args):
    """Rename a speaker."""
    manager = SpeakerFingerprintManager(db_path=args.db_path)
    
    # Find speaker by name or ID
    speakers = manager.list_speakers()
    target_speaker = None
    
    for speaker in speakers:
        if speaker['name'] == args.speaker or speaker['id'].startswith(args.speaker):
            target_speaker = speaker
            break
    
    if not target_speaker:
        print(f"Speaker '{args.speaker}' not found.")
        return 1
    
    success = manager.rename_speaker(target_speaker['id'], args.new_name)
    
    if success:
        print(f"Renamed '{target_speaker['name']}' to '{args.new_name}'")
    else:
        print("Failed to rename speaker.")
        return 1


def merge_speakers(args):
    """Merge multiple speakers into one."""
    manager = SpeakerFingerprintManager(db_path=args.db_path)
    
    # Find all specified speakers
    speakers = manager.list_speakers()
    speaker_ids = []
    speaker_names = []
    
    for speaker_query in args.speakers:
        found = False
        for speaker in speakers:
            if speaker['name'] == speaker_query or speaker['id'].startswith(speaker_query):
                speaker_ids.append(speaker['id'])
                speaker_names.append(speaker['name'])
                found = True
                break
        
        if not found:
            print(f"Speaker '{speaker_query}' not found.")
            return 1
    
    if len(speaker_ids) < 2:
        print("Need at least 2 speakers to merge.")
        return 1
    
    # Confirm merge
    print(f"Will merge speakers: {', '.join(speaker_names)}")
    if args.target_name:
        print(f"Into: {args.target_name}")
    else:
        print(f"Into: {speaker_names[0]} (first speaker)")
    
    if not args.yes:
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Perform merge
    result_id = manager.merge_speakers(speaker_ids, args.target_name)
    
    if result_id:
        print("Speakers merged successfully.")
    else:
        print("Failed to merge speakers.")
        return 1


def export_database(args):
    """Export speaker database."""
    manager = SpeakerFingerprintManager(db_path=args.db_path)
    
    export_path = Path(args.output)
    success = manager.export_database(export_path)
    
    if success:
        print(f"Database exported to: {export_path}")
    else:
        print("Failed to export database.")
        return 1


def import_database(args):
    """Import speaker database."""
    manager = SpeakerFingerprintManager(db_path=args.db_path)
    
    import_path = Path(args.input)
    if not import_path.exists():
        print(f"Import file not found: {import_path}")
        return 1
    
    # Confirm if not merging (will replace)
    if not args.merge and not args.yes:
        response = input("This will REPLACE the existing database. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    count = manager.import_database(import_path, merge=args.merge)
    
    if count > 0:
        print(f"Imported {count} speakers.")
    else:
        print("No speakers imported.")


def verify_speaker(args):
    """Verify a speaker from an audio file."""
    manager = SpeakerFingerprintManager(db_path=args.db_path)
    
    # Extract embedding from audio
    embedding = manager.extract_embedding(
        args.audio_file,
        start_time=args.start,
        end_time=args.end
    )
    
    if embedding is None:
        print("Failed to extract embedding from audio.")
        return 1
    
    # Identify speaker
    result = manager.identify_speaker(embedding, return_all_matches=True)
    
    if result['identified']:
        print(f"\nIdentified as: {result['speaker_name']}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        if result.get('all_matches') and len(result['all_matches']) > 1:
            print("\nOther potential matches:")
            for match in result['all_matches'][1:]:
                print(f"  - {match['speaker_name']} ({match['confidence']:.1%})")
    else:
        print(f"\nNo match found (best confidence: {result['confidence']:.1%})")
        print(f"Threshold: {manager.similarity_threshold:.1%}")


def stats(args):
    """Show database statistics."""
    manager = SpeakerFingerprintManager(db_path=args.db_path)
    
    speakers = manager.list_speakers()
    
    if not speakers:
        print("No speakers in database.")
        return
    
    # Calculate statistics
    total_samples = sum(s['sample_count'] for s in speakers)
    
    print("\nSPEAKER DATABASE STATISTICS")
    print("=" * 40)
    print(f"Total speakers: {len(speakers)}")
    print(f"Total samples: {total_samples}")
    print(f"Average samples per speaker: {total_samples / len(speakers):.1f}")
    print(f"Database location: {manager.db_path}")
    
    # Find speakers with most samples
    top_speakers = sorted(speakers, key=lambda x: x['sample_count'], reverse=True)[:5]
    
    print("\nTop speakers by sample count:")
    for speaker in top_speakers:
        print(f"  - {speaker['name']}: {speaker['sample_count']} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Manage speaker database for voice recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all speakers
  python speaker_cli.py list
  
  # Rename a speaker
  python speaker_cli.py rename "Speaker_1" "John Doe"
  
  # Merge speakers
  python speaker_cli.py merge Speaker_1 Speaker_2 --target "John Doe"
  
  # Verify speaker from audio
  python speaker_cli.py verify audio.wav --start 10 --end 30
  
  # Export/import database
  python speaker_cli.py export speakers_backup.json
  python speaker_cli.py import speakers_backup.json --merge
        """
    )
    
    # Global options
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Custom database path (default: from config)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all speakers")
    
    # Rename command
    rename_parser = subparsers.add_parser("rename", help="Rename a speaker")
    rename_parser.add_argument("speaker", help="Speaker name or ID prefix")
    rename_parser.add_argument("new_name", help="New name for speaker")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple speakers")
    merge_parser.add_argument("speakers", nargs="+", help="Speaker names or IDs to merge")
    merge_parser.add_argument("--target", help="Target name for merged speaker")
    merge_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export database")
    export_parser.add_argument("output", help="Output file path")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import database")
    import_parser.add_argument("input", help="Input file path")
    import_parser.add_argument("--merge", action="store_true", help="Merge with existing database")
    import_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify speaker from audio")
    verify_parser.add_argument("audio_file", help="Audio file path")
    verify_parser.add_argument("--start", type=float, help="Start time in seconds")
    verify_parser.add_argument("--end", type=float, help="End time in seconds")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup config
    Config.setup_directories()
    
    # Execute command
    try:
        if args.command == "list":
            return list_speakers(args) or 0
        elif args.command == "rename":
            return rename_speaker(args) or 0
        elif args.command == "merge":
            return merge_speakers(args) or 0
        elif args.command == "export":
            return export_database(args) or 0
        elif args.command == "import":
            return import_database(args) or 0
        elif args.command == "verify":
            return verify_speaker(args) or 0
        elif args.command == "stats":
            return stats(args) or 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())