# File: speaker_id/fingerprint_manager.py
"""
Speaker Fingerprinting Manager - Core module for speaker recognition
Handles embedding extraction, database management, and speaker matching
Now supports multiple voice samples per speaker for improved accuracy
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import hashlib
import uuid

# Core dependencies
import torch
import torchaudio
from scipy.spatial.distance import cosine
import chromadb

# Speaker embedding model
try:
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("Warning: SpeechBrain not installed. Speaker recognition will not be available.")

from core.config import Config


class SpeakerFingerprintManager:
    """Manages speaker embeddings and recognition using ChromaDB with multi-sample support."""
    
    def __init__(self, db_path: Optional[Path] = None, 
                 similarity_threshold: float = 0.85,
                 device: Optional[str] = None,
                 max_samples_per_speaker: int = 15):
        """
        Initialize the speaker fingerprint manager.
        
        Args:
            db_path: Path to ChromaDB storage (defaults to Config.SPEAKER_DB_DIR)
            similarity_threshold: Minimum cosine similarity for speaker match (0-1)
            device: Device for model inference ('cuda' or 'cpu')
            max_samples_per_speaker: Maximum voice samples to store per speaker
        """
        self.db_path = db_path or Config.SPEAKER_DB_DIR
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.device = device or Config.WHISPER_DEVICE
        self.max_samples_per_speaker = Config.SPEAKER_MAX_SAMPLES or max_samples_per_speaker

        # Initialize ChromaDB
        self._init_database()
        
        # Load speaker embedding model
        self._load_embedding_model()
        
        # Load metadata
        self.metadata_file = self.db_path / "speaker_metadata.json"
        self.metadata = self._load_metadata()
        
        # Migrate old database format if needed
        self._migrate_database_if_needed()
    
    def _init_database(self):
        """Initialize ChromaDB client and collection."""
        # Use new ChromaDB API (v0.4+)
        import chromadb
        
        # Configure ChromaDB with persistent storage using new API
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path / "chroma_db")
        )
        
        # Create or get speaker collection
        try:
            self.collection = self.chroma_client.get_collection("speaker_embeddings_v2")
        except:
            self.collection = self.chroma_client.create_collection(
                name="speaker_embeddings_v2",
                metadata={"description": "Multi-sample speaker voice embeddings for identification"}
            )
    
    def _migrate_database_if_needed(self):
        """Migrate from old single-sample format to multi-sample format."""
        try:
            # Check if old collection exists
            old_collection = None
            try:
                old_collection = self.chroma_client.get_collection("speaker_embeddings")
            except:
                return  # No old collection to migrate
            
            if old_collection and old_collection.count() > 0:
                print("Migrating speaker database to multi-sample format...")
                
                # Get all old entries
                old_data = old_collection.get(include=['embeddings', 'metadatas'])
                
                migrated_count = 0
                for old_id, embedding, metadata in zip(old_data['ids'], 
                                                      old_data['embeddings'], 
                                                      old_data['metadatas']):
                    # Create new entry with sample format
                    new_id = str(uuid.uuid4())
                    new_metadata = {
                        "speaker_id": old_id,
                        "speaker_name": metadata.get('name', 'Unknown'),
                        "sample_index": 1,
                        "sample_timestamp": metadata.get('last_seen', datetime.now().isoformat()),
                        "quality_score": 1.0,
                        **metadata
                    }
                    
                    self.collection.add(
                        embeddings=[embedding],
                        ids=[new_id],
                        metadatas=[new_metadata]
                    )
                    migrated_count += 1
                
                # Delete old collection
                self.chroma_client.delete_collection("speaker_embeddings")
                print(f"Migration complete: {migrated_count} speakers migrated to multi-sample format")
                
        except Exception as e:
            print(f"Warning: Database migration failed: {e}")
    
    def _load_embedding_model(self):
        """Load the SpeechBrain ECAPA-TDNN model for speaker embeddings."""
        if not SPEECHBRAIN_AVAILABLE:
            self.embedding_model = None
            return
        
        try:
            # Load pre-trained ECAPA-TDNN model
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            self.embedding_model.eval()
            print(f"Loaded speaker embedding model on {self.device}")
        except Exception as e:
            print(f"Error loading speaker embedding model: {e}")
            self.embedding_model = None
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load speaker metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "speakers": {},
            "aliases": {},
            "stats": {
                "total_speakers": 0,
                "total_embeddings": 0,
                "last_updated": None
            },
            "version": "2.0"  # Multi-sample version
        }
    
    def _save_metadata(self):
        """Save speaker metadata to JSON file."""
        self.metadata["stats"]["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
    
    def extract_embedding(self, audio_path: str, start_time: float = None, 
                         end_time: float = None) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio segment.
        
        Args:
            audio_path: Path to audio file or video file
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            Embedding vector as numpy array or None if extraction fails
        """
        if not self.embedding_model:
            return None
        
        import tempfile
        import shutil
        import subprocess
        import sys
        
        # Supported video extensions
        video_exts = ['.mp4', '.mkv', '.mov', '.avi']
        file_ext = os.path.splitext(audio_path)[1].lower()
        temp_audio_path = None
        try:
            # If input is a video, extract audio to temp wav file
            if file_ext in video_exts:
                temp_dir = tempfile.mkdtemp()
                temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', audio_path, '-vn',
                    '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio_path
                ]
                result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if not os.path.exists(temp_audio_path):
                    print(f"Error extracting audio from video: {result.stderr.decode(errors='ignore')}")
                    return None
                audio_path_to_use = temp_audio_path
            else:
                audio_path_to_use = audio_path

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path_to_use)
            
            # Extract segment if times provided
            if start_time is not None and end_time is not None:
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                waveform = waveform[:, start_sample:end_sample]
            
            # Resample if necessary (model expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
        finally:
            # Clean up temp audio file if created
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    shutil.rmtree(os.path.dirname(temp_audio_path))
                except Exception:
                    pass
    
    def extract_embeddings_from_segments(self, audio_path: str, 
                                          segments: List[Dict], 
                                          max_segments_per_speaker: int = 10,
                                          min_segment_duration: float = 2.0) -> Dict[str, List[np.ndarray]]:
        """
        Extract multiple embeddings for each speaker from segments.
        
        Args:
            audio_path: Path to audio file or video file
            segments: List of segment dicts with 'speaker', 'start', 'end'
            max_segments_per_speaker: Maximum segments to process per speaker
            min_segment_duration: Minimum segment duration in seconds
            
        Returns:
            Dict mapping speaker labels to lists of embeddings
        """
        if not self.embedding_model:
            return {}
        
        import tempfile
        import shutil
        import subprocess
        
        print(f"Processing {len(segments)} segments for speaker recognition...")
        
        # Check if input is video and extract audio if needed
        video_exts = ['.mp4', '.mkv', '.mov', '.avi']
        file_ext = os.path.splitext(audio_path)[1].lower()
        temp_audio_path = None
        
        try:
            # If input is a video, extract audio to temp wav file
            if file_ext in video_exts:
                print("Extracting audio from video file...")
                temp_dir = tempfile.mkdtemp()
                temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
                
                # Use ffmpeg to extract audio
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', audio_path, '-vn',
                    '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    '-threads', '0',  # Use all available CPU threads
                    temp_audio_path
                ]
                
                # Run ffmpeg
                result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                    error_msg = result.stderr.decode(errors='ignore')
                    print(f"Error extracting audio from video: {error_msg}")
                    return {}
                
                audio_path_to_use = temp_audio_path
            else:
                audio_path_to_use = audio_path
            
            # Load audio file ONCE
            print("Loading audio file...")
            waveform, sample_rate = torchaudio.load(audio_path_to_use)
            
            # Resample if necessary (model expects 16kHz)
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate}Hz to 16000Hz...")
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            speaker_embeddings = {}
            speaker_segments = {}
            
            # Group segments by speaker and filter
            for segment in segments:
                speaker = segment.get('speaker', 'Unknown')
                duration = segment.get('end', 0) - segment.get('start', 0)
                
                # Skip very short segments
                if duration < min_segment_duration:
                    continue
                    
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(segment)
            
            print(f"Found {len(speaker_segments)} unique speakers")
            
            # Process each speaker
            for speaker, segs in speaker_segments.items():
                print(f"Processing {speaker}: {len(segs)} segments available")
                
                # Sample segments if too many
                if len(segs) > max_segments_per_speaker:
                    # Take evenly distributed segments
                    indices = np.linspace(0, len(segs) - 1, max_segments_per_speaker, dtype=int)
                    sampled_segs = [segs[i] for i in indices]
                    print(f"  Sampling {len(sampled_segs)} segments")
                else:
                    sampled_segs = segs
                
                embeddings = []
                
                # Extract embeddings for sampled segments
                for seg in sampled_segs:
                    try:
                        # Extract segment from pre-loaded waveform
                        start_sample = int(seg.get('start') * sample_rate)
                        end_sample = int(seg.get('end') * sample_rate)
                        
                        # Ensure we don't go out of bounds
                        start_sample = max(0, start_sample)
                        end_sample = min(waveform.shape[1], end_sample)
                        
                        segment_waveform = waveform[:, start_sample:end_sample]
                        
                        # Skip if segment is too short
                        if segment_waveform.shape[1] < sample_rate * min_segment_duration:
                            continue
                        
                        # Extract embedding
                        with torch.no_grad():
                            embedding = self.embedding_model.encode_batch(segment_waveform)
                            embedding = embedding.squeeze().cpu().numpy()
                        
                        # Normalize embedding
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding)
                        
                    except Exception as e:
                        print(f"  Warning: Failed to process segment: {e}")
                        continue
                
                if embeddings:
                    speaker_embeddings[speaker] = embeddings
                    print(f"  Created {len(embeddings)} embeddings")
                else:
                    print(f"  Warning: No valid embeddings for {speaker}")
            
            return speaker_embeddings
            
        finally:
            # Clean up temp audio file if created
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    shutil.rmtree(os.path.dirname(temp_audio_path))
                except Exception as e:
                    print(f"Warning: Could not clean up temp file: {e}")
    
    def identify_speaker(self, embedding: np.ndarray, 
                        return_all_matches: bool = False) -> Dict[str, Any]:
        """
        Identify speaker by comparing embedding to database using multi-sample matching.
        
        Args:
            embedding: Speaker embedding vector
            return_all_matches: Return all potential matches, not just best
            
        Returns:
            Dict with identification results
        """
        if self.collection.count() == 0:
            return {
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "confidence": 0.0,
                "message": "No speakers in database"
            }
        
        # Query ChromaDB for similar embeddings
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(50, self.collection.count())  # Get more results for multi-sample
        )
        
        if not results['ids'][0]:
            return {
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "confidence": 0.0,
                "message": "No matches found"
            }
        
        # Group results by speaker
        speaker_scores = {}
        speaker_info = {}
        
        for i, (sample_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            metadata = results['metadatas'][0][i]
            speaker_id = metadata.get('speaker_id')
            speaker_name = metadata.get('speaker_name', 'Unknown')
            
            # Convert distance to similarity
            similarity = 1 - (distance / 2)  # Cosine distance range is [0, 2]
            
            if speaker_id not in speaker_scores:
                speaker_scores[speaker_id] = []
                speaker_info[speaker_id] = {
                    'name': speaker_name,
                    'metadata': metadata
                }
            
            speaker_scores[speaker_id].append(similarity)
        
        # Calculate aggregate scores for each speaker
        speaker_results = []
        for speaker_id, scores in speaker_scores.items():
            # Use top-k averaging (average top 3 scores if available)
            top_scores = sorted(scores, reverse=True)[:3]
            avg_score = np.mean(top_scores)
            
            if avg_score >= self.similarity_threshold:
                speaker_results.append({
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_info[speaker_id]['name'],
                    "confidence": avg_score,
                    "match_count": len(scores),
                    "max_score": max(scores)
                })
        
        # Sort by confidence
        speaker_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        if speaker_results:
            best_match = speaker_results[0]
            return {
                "identified": True,
                "speaker_id": best_match["speaker_id"],
                "speaker_name": best_match["speaker_name"],
                "confidence": best_match["confidence"],
                "match_count": best_match["match_count"],
                "all_matches": speaker_results if return_all_matches else None
            }
        else:
            # Return best match even if below threshold
            if speaker_scores:
                best_speaker = max(speaker_scores.items(), 
                                 key=lambda x: max(x[1]))[0]
                best_score = max(speaker_scores[best_speaker])
                
                return {
                    "identified": False,
                    "speaker_id": None,
                    "speaker_name": None,
                    "confidence": best_score,
                    "message": f"Best match below threshold ({self.similarity_threshold})"
                }
            else:
                return {
                    "identified": False,
                    "speaker_id": None,
                    "speaker_name": None,
                    "confidence": 0.0,
                    "message": "No matches found"
                }
    
    def add_speaker(self, name: str, embedding: np.ndarray, 
                   metadata: Optional[Dict] = None) -> str:
        """
        Add a new speaker to the database with first sample.
        
        Args:
            name: Speaker name
            embedding: Speaker embedding vector
            metadata: Additional metadata
            
        Returns:
            Speaker ID
        """
        # Generate unique speaker ID
        speaker_id = self._generate_speaker_id(name)
        
        # Add first sample
        self.add_speaker_sample(speaker_id, name, embedding, metadata)
        
        # Update metadata file
        self.metadata["speakers"][speaker_id] = {
            "name": name,
            "created": datetime.now().isoformat(),
            "sample_count": 1
        }
        self.metadata["stats"]["total_speakers"] += 1
        self._save_metadata()
        
        print(f"Added new speaker: {name} (ID: {speaker_id})")
        return speaker_id
    
    def add_speaker_sample(self, speaker_id: str, speaker_name: str,
                          embedding: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Add a new sample for an existing speaker.
        
        Args:
            speaker_id: Speaker ID
            speaker_name: Speaker name
            embedding: New embedding vector
            metadata: Additional metadata for this sample
            
        Returns:
            Success status
        """
        try:
            # Check current sample count
            current_samples = self._get_speaker_samples(speaker_id)
            
            # Generate unique sample ID
            sample_id = str(uuid.uuid4())
            
            # Prepare sample metadata
            now = datetime.now().isoformat()
            sample_metadata = {
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "sample_index": len(current_samples) + 1,
                "sample_timestamp": now,
                "quality_score": 1.0,  # Could be computed based on audio quality
                **(metadata or {})
            }
            
            # If at max samples, remove oldest
            if len(current_samples) >= self.max_samples_per_speaker:
                # Sort by timestamp and remove oldest
                sorted_samples = sorted(current_samples, 
                                      key=lambda x: x['metadata'].get('sample_timestamp', ''))
                oldest_id = sorted_samples[0]['id']
                self.collection.delete(ids=[oldest_id])
                print(f"  Removed oldest sample to maintain limit of {self.max_samples_per_speaker}")
            
            # Add new sample
            self.collection.add(
                embeddings=[embedding.tolist()],
                ids=[sample_id],
                metadatas=[sample_metadata]
            )
            
            # Update metadata
            self.metadata["stats"]["total_embeddings"] = self.collection.count()
            if speaker_id in self.metadata["speakers"]:
                self.metadata["speakers"][speaker_id]["last_updated"] = now
                self.metadata["speakers"][speaker_id]["sample_count"] = min(
                    len(current_samples) + 1, 
                    self.max_samples_per_speaker
                )
            self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error adding speaker sample: {e}")
            return False
    
    def _get_speaker_samples(self, speaker_id: str) -> List[Dict]:
        """Get all samples for a specific speaker."""
        try:
            # Query all entries for this speaker
            results = self.collection.get(
                where={"speaker_id": speaker_id},
                include=['embeddings', 'metadatas']
            )
            
            samples = []
            for i, (sample_id, embedding) in enumerate(zip(results['ids'], results['embeddings'])):
                samples.append({
                    'id': sample_id,
                    'embedding': np.array(embedding),
                    'metadata': results['metadatas'][i]
                })
            
            return samples
            
        except Exception:
            return []
    
    def update_speaker_embedding(self, speaker_id: str, 
                               new_embedding: np.ndarray,
                               metadata: Optional[Dict] = None) -> bool:
        """
        Add new embedding sample for speaker (replaces old weighted average approach).
        
        Args:
            speaker_id: Speaker ID
            new_embedding: New embedding vector
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        # Get speaker name from existing samples
        samples = self._get_speaker_samples(speaker_id)
        if not samples:
            return False
        
        speaker_name = samples[0]['metadata'].get('speaker_name', 'Unknown')
        
        # Add as new sample
        return self.add_speaker_sample(speaker_id, speaker_name, new_embedding, metadata)
    
    def process_diarized_segments(self, audio_path: str, segments: List[Dict],
                            auto_add_unknown: bool = True) -> List[Dict]:
        """
        Process diarized segments and identify speakers using multi-sample approach.
        FIXED: Each diarized speaker is independently evaluated against ALL database speakers.
        
        Args:
            audio_path: Path to audio file
            segments: List of diarized segments
            auto_add_unknown: Automatically add unidentified speakers
            
        Returns:
            Updated segments with identified speakers
        """
        print(f"\nMulti-sample speaker recognition for {len(segments)} segments")
        start_time = datetime.now()
        
        # Extract multiple embeddings per speaker
        speaker_embeddings = self.extract_embeddings_from_segments(
            audio_path, 
            segments,
            max_segments_per_speaker=10,
            min_segment_duration=2.0
        )
        
        print(f"Extracted embeddings in {(datetime.now() - start_time).seconds}s")
        
        # Track speaker mappings and unmatched speakers
        speaker_mapping = {}
        unmatched_speakers = set()
        
        # Process each diarized speaker independently
        for original_speaker, embeddings in speaker_embeddings.items():
            print(f"\nProcessing {original_speaker} with {len(embeddings)} embeddings...")
            
            # Try to identify using each embedding and aggregate results
            all_results = []
            for embedding in embeddings:
                result = self.identify_speaker(embedding, return_all_matches=True)
                if result['identified'] or result.get('all_matches'):
                    all_results.append(result)
            
            if all_results:
                # Aggregate identification results across ALL database speakers
                # Don't exclude any speakers that have already been matched
                speaker_votes = {}
                
                for result in all_results:
                    # Consider all matches, not just the best one
                    matches_to_consider = []
                    
                    if result['identified']:
                        # Add the primary match
                        matches_to_consider.append({
                            'speaker_id': result['speaker_id'],
                            'speaker_name': result['speaker_name'],
                            'confidence': result['confidence']
                        })
                    
                    # Also consider all other matches
                    if result.get('all_matches'):
                        matches_to_consider.extend(result['all_matches'])
                    
                    # Vote for each potential match
                    for match in matches_to_consider:
                        speaker_id = match['speaker_id']
                        speaker_name = match['speaker_name']
                        confidence = match['confidence']
                        
                        if speaker_id not in speaker_votes:
                            speaker_votes[speaker_id] = {
                                'name': speaker_name,
                                'scores': [],
                                'count': 0
                            }
                        
                        speaker_votes[speaker_id]['scores'].append(confidence)
                        speaker_votes[speaker_id]['count'] += 1
                
                # Find best match by voting and average confidence
                if speaker_votes:
                    # Sort by: (1) number of votes, (2) average confidence
                    sorted_candidates = sorted(
                        speaker_votes.items(),
                        key=lambda x: (x[1]['count'], np.mean(x[1]['scores'])),
                        reverse=True
                    )
                    
                    # Check if best candidate meets threshold
                    best_speaker_id, best_data = sorted_candidates[0]
                    avg_confidence = np.mean(best_data['scores'])
                    vote_percentage = best_data['count'] / len(all_results)
                    
                    # Log all candidates for debugging
                    print(f"  Candidates for {original_speaker}:")
                    for speaker_id, data in sorted_candidates[:3]:
                        avg_conf = np.mean(data['scores'])
                        vote_pct = data['count'] / len(all_results) * 100
                        print(f"    - {data['name']}: {avg_conf:.1%} confidence, {vote_pct:.0f}% votes")
                    
                    # Use weighted decision: high confidence OR high vote percentage
                    if avg_confidence >= self.similarity_threshold or \
                    (avg_confidence >= self.similarity_threshold * 0.9 and vote_percentage >= 0.7):
                        # Known speaker identified
                        speaker_mapping[original_speaker] = {
                            "id": best_speaker_id,
                            "name": best_data['name'],
                            "confidence": avg_confidence,
                            "match_count": best_data['count'],
                            "vote_percentage": vote_percentage
                        }
                        
                        # Add new samples for this speaker
                        for embedding in embeddings[:3]:  # Add up to 3 best samples
                            self.add_speaker_sample(best_speaker_id, best_data['name'], embedding)
                        
                        print(f"  ✓ Identified as: {best_data['name']} "
                            f"(confidence: {avg_confidence:.1%}, votes: {vote_percentage:.0%})")
                    else:
                        # Below threshold
                        unmatched_speakers.add(original_speaker)
                        print(f"  ✗ No confident match (best: {best_data['name']} at {avg_confidence:.1%})")
                else:
                    unmatched_speakers.add(original_speaker)
                    print(f"  ✗ No matches found")
            else:
                unmatched_speakers.add(original_speaker)
                print(f"  ✗ No successful identifications")
        
        # Second pass: Check unmatched speakers with slightly lower threshold
        if unmatched_speakers:
            print(f"\nSecond pass for {len(unmatched_speakers)} unmatched speakers...")
            second_pass_threshold = self.similarity_threshold * 0.85
            
            for original_speaker in list(unmatched_speakers):
                if original_speaker not in speaker_embeddings:
                    continue
                    
                embeddings = speaker_embeddings[original_speaker]
                print(f"  Re-checking {original_speaker} with threshold {second_pass_threshold:.1%}...")
                
                # Temporarily lower threshold
                original_threshold = self.similarity_threshold
                self.similarity_threshold = second_pass_threshold
                
                try:
                    # Re-evaluate with lower threshold
                    all_results = []
                    for embedding in embeddings:
                        result = self.identify_speaker(embedding, return_all_matches=True)
                        if result['confidence'] >= second_pass_threshold:
                            all_results.append(result)
                    
                    if all_results:
                        # Find most common match
                        speaker_votes = {}
                        for result in all_results:
                            if result.get('speaker_id'):
                                speaker_id = result['speaker_id']
                                if speaker_id not in speaker_votes:
                                    speaker_votes[speaker_id] = {
                                        'name': result['speaker_name'],
                                        'scores': []
                                    }
                                speaker_votes[speaker_id]['scores'].append(result['confidence'])
                        
                        if speaker_votes:
                            best_speaker_id = max(speaker_votes.items(),
                                                key=lambda x: (len(x[1]['scores']), np.mean(x[1]['scores'])))[0]
                            best_data = speaker_votes[best_speaker_id]
                            avg_confidence = np.mean(best_data['scores'])
                            
                            speaker_mapping[original_speaker] = {
                                "id": best_speaker_id,
                                "name": best_data['name'],
                                "confidence": avg_confidence,
                                "match_count": len(best_data['scores']),
                                "second_pass": True
                            }
                            
                            # Add samples
                            for embedding in embeddings[:2]:
                                self.add_speaker_sample(best_speaker_id, best_data['name'], embedding)
                            
                            unmatched_speakers.remove(original_speaker)
                            print(f"    ✓ Matched on second pass: {best_data['name']} ({avg_confidence:.1%})")
                            
                finally:
                    # Restore original threshold
                    self.similarity_threshold = original_threshold
        
        # Handle remaining unknown speakers
        for original_speaker in unmatched_speakers:
            if original_speaker in speaker_embeddings:
                embeddings = speaker_embeddings[original_speaker]
                
                if auto_add_unknown and embeddings:
                    # Generate name for unknown speaker
                    unknown_name = f"Speaker_{len(self.metadata['speakers']) + 1}"
                    
                    # Add with first embedding
                    speaker_id = self.add_speaker(unknown_name, embeddings[0])
                    
                    # Add additional samples
                    for embedding in embeddings[1:min(len(embeddings), 5)]:
                        self.add_speaker_sample(speaker_id, unknown_name, embedding)
                    
                    speaker_mapping[original_speaker] = {
                        "id": speaker_id,
                        "name": unknown_name,
                        "confidence": 1.0,
                        "is_new": True
                    }
                    
                    print(f"\n  Added as new speaker: {unknown_name}")
                else:
                    speaker_mapping[original_speaker] = {
                        "id": None,
                        "name": original_speaker,
                        "confidence": 0.0
                    }
        
        # Update segments with identified speakers
        print("\nUpdating segment labels...")
        updated_segments = []
        for segment in segments:
            updated_segment = segment.copy()
            original_speaker = segment.get('speaker', 'Unknown')
            
            if original_speaker in speaker_mapping and speaker_mapping[original_speaker]:
                mapping = speaker_mapping[original_speaker]
                updated_segment['speaker'] = mapping['name']
                updated_segment['speaker_id'] = mapping['id']
                updated_segment['speaker_confidence'] = mapping['confidence']
                if mapping.get('is_new'):
                    updated_segment['speaker_is_new'] = True
                if mapping.get('second_pass'):
                    updated_segment['speaker_second_pass'] = True
            
            updated_segments.append(updated_segment)
        
        # Print summary
        total_time = (datetime.now() - start_time).seconds
        print(f"\nSpeaker recognition completed in {total_time}s")
        
        # Summary statistics
        identified_count = len([m for m in speaker_mapping.values() if m.get('id') and not m.get('is_new')])
        new_count = len([m for m in speaker_mapping.values() if m.get('is_new')])
        unidentified_count = len([m for m in speaker_mapping.values() if not m.get('id')])
        
        print(f"Results: {identified_count} identified, {new_count} new, {unidentified_count} unidentified")
        
        return updated_segments
    
    def _generate_speaker_id(self, name: str) -> str:
        """Generate unique speaker ID."""
        timestamp = datetime.now().isoformat()
        unique_string = f"{name}_{timestamp}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def list_speakers(self) -> List[Dict[str, Any]]:
        """List all speakers in the database with accurate sample counts."""
        if self.collection.count() == 0:
            return []
        
        # Get all samples and group by speaker
        results = self.collection.get(include=['metadatas'])
        
        speakers = {}
        for i, (sample_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
            speaker_id = metadata.get('speaker_id')
            speaker_name = metadata.get('speaker_name', 'Unknown')
            
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    "id": speaker_id,
                    "name": speaker_name,
                    "first_seen": metadata.get('sample_timestamp'),
                    "last_seen": metadata.get('sample_timestamp'),
                    "sample_count": 0,
                    "samples": []
                }
            
            speakers[speaker_id]['sample_count'] += 1
            speakers[speaker_id]['samples'].append({
                'id': sample_id,
                'timestamp': metadata.get('sample_timestamp')
            })
            
            # Update first/last seen
            timestamp = metadata.get('sample_timestamp', '')
            if timestamp:
                if not speakers[speaker_id]['first_seen'] or timestamp < speakers[speaker_id]['first_seen']:
                    speakers[speaker_id]['first_seen'] = timestamp
                if not speakers[speaker_id]['last_seen'] or timestamp > speakers[speaker_id]['last_seen']:
                    speakers[speaker_id]['last_seen'] = timestamp
        
        # Convert to list and clean up
        speaker_list = []
        for speaker_data in speakers.values():
            speaker_list.append({
                "id": speaker_data['id'],
                "name": speaker_data['name'],
                "first_seen": speaker_data['first_seen'],
                "last_seen": speaker_data['last_seen'],
                "sample_count": speaker_data['sample_count']
            })
        
        return sorted(speaker_list, key=lambda x: x['name'])
    
    def rename_speaker(self, speaker_id: str, new_name: str) -> bool:
        """Rename a speaker (updates all samples)."""
        try:
            # Get all samples for this speaker
            samples = self._get_speaker_samples(speaker_id)
            
            if not samples:
                return False
            
            # Update each sample
            for sample in samples:
                sample['metadata']['speaker_name'] = new_name
                self.collection.update(
                    ids=[sample['id']],
                    metadatas=[sample['metadata']]
                )
            
            # Update metadata file
            if speaker_id in self.metadata["speakers"]:
                self.metadata["speakers"][speaker_id]["name"] = new_name
                self._save_metadata()
            
            print(f"Renamed speaker to: {new_name} (updated {len(samples)} samples)")
            return True
            
        except Exception as e:
            print(f"Error renaming speaker: {e}")
            return False
    
    def merge_speakers(self, speaker_ids: List[str], 
                      target_name: Optional[str] = None) -> Optional[str]:
        """
        Merge multiple speakers into one (combines all samples).
        
        Args:
            speaker_ids: List of speaker IDs to merge
            target_name: Name for merged speaker (uses first speaker's name if None)
            
        Returns:
            Merged speaker ID or None if failed
        """
        if len(speaker_ids) < 2:
            return None
        
        try:
            # Get all samples from all speakers
            all_samples = []
            speaker_names = []
            
            for speaker_id in speaker_ids:
                samples = self._get_speaker_samples(speaker_id)
                all_samples.extend(samples)
                if samples:
                    speaker_names.append(samples[0]['metadata'].get('speaker_name', 'Unknown'))
            
            if not all_samples:
                return None
            
            # Use target name or first speaker's name
            if not target_name:
                target_name = speaker_names[0] if speaker_names else 'Merged Speaker'
            
            # Use first speaker ID as target
            target_id = speaker_ids[0]
            
            # Sort all samples by timestamp
            all_samples.sort(key=lambda x: x['metadata'].get('sample_timestamp', ''))
            
            # Keep only the most recent samples up to max limit
            samples_to_keep = all_samples[-self.max_samples_per_speaker:]
            samples_to_delete = all_samples[:-self.max_samples_per_speaker]
            
            # Delete excess samples
            if samples_to_delete:
                delete_ids = [s['id'] for s in samples_to_delete]
                self.collection.delete(ids=delete_ids)
            
            # Update remaining samples to point to target speaker
            for sample in samples_to_keep:
                sample['metadata']['speaker_id'] = target_id
                sample['metadata']['speaker_name'] = target_name
                self.collection.update(
                    ids=[sample['id']],
                    metadatas=[sample['metadata']]
                )
            
            # Delete samples from other speakers (they've been reassigned)
            for speaker_id in speaker_ids[1:]:
                # Samples already reassigned, just clean up metadata
                if speaker_id in self.metadata["speakers"]:
                    del self.metadata["speakers"][speaker_id]
            
            # Update target speaker metadata
            self.metadata["speakers"][target_id]["name"] = target_name
            self.metadata["speakers"][target_id]["sample_count"] = len(samples_to_keep)
            self.metadata["stats"]["total_speakers"] = len(self.metadata["speakers"])
            self._save_metadata()
            
            print(f"Merged {len(speaker_ids)} speakers into: {target_name}")
            print(f"  Total samples: {len(all_samples)} -> {len(samples_to_keep)} kept")
            return target_id
            
        except Exception as e:
            print(f"Error merging speakers: {e}")
            return None
    
    def export_database(self, export_path: Path) -> bool:
        """Export speaker database for backup or sharing."""
        try:
            export_data = {
                "version": "2.0",  # Multi-sample version
                "exported_at": datetime.now().isoformat(),
                "speakers": [],
                "config": {
                    "max_samples_per_speaker": self.max_samples_per_speaker,
                    "similarity_threshold": self.similarity_threshold
                }
            }
            
            # Get all samples
            results = self.collection.get(include=['embeddings', 'metadatas'])
            
            # Group by speaker
            speakers = {}
            for sample_id, embedding, metadata in zip(results['ids'], 
                                                    results['embeddings'], 
                                                    results['metadatas']):
                speaker_id = metadata.get('speaker_id')
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        "id": speaker_id,
                        "name": metadata.get('speaker_name', 'Unknown'),
                        "samples": []
                    }
                
                speakers[speaker_id]['samples'].append({
                    "id": sample_id,
                    "embedding": embedding,
                    "metadata": metadata
                })
            
            # Add to export data
            for speaker_data in speakers.values():
                export_data["speakers"].append(speaker_data)
            
            # Save to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            total_samples = sum(len(s['samples']) for s in export_data['speakers'])
            print(f"Exported {len(export_data['speakers'])} speakers "
                  f"({total_samples} total samples) to {export_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting database: {e}")
            return False
    
    def import_database(self, import_path: Path, merge: bool = True) -> int:
        """
        Import speaker database from file.
        
        Args:
            import_path: Path to import file
            merge: Merge with existing database (False = replace)
            
        Returns:
            Number of speakers imported
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Check version
            version = import_data.get('version', '1.0')
            
            if not merge:
                # Clear existing database
                all_ids = self.collection.get()['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
                self.metadata = self._load_metadata()
            
            imported_speakers = 0
            imported_samples = 0
            
            if version == '2.0':
                # Multi-sample format
                for speaker_data in import_data["speakers"]:
                    speaker_id = speaker_data['id']
                    speaker_name = speaker_data['name']
                    
                    # Check if speaker exists
                    existing_samples = self._get_speaker_samples(speaker_id) if merge else []
                    
                    if not existing_samples:
                        # New speaker
                        imported_speakers += 1
                        
                        # Add metadata
                        self.metadata["speakers"][speaker_id] = {
                            "name": speaker_name,
                            "imported_at": datetime.now().isoformat()
                        }
                    
                    # Import samples
                    for sample in speaker_data['samples']:
                        if merge and existing_samples:
                            # Check if we should add this sample
                            if len(existing_samples) >= self.max_samples_per_speaker:
                                continue
                        
                        self.collection.add(
                            ids=[sample['id']],
                            embeddings=[sample['embedding']],
                            metadatas=[sample['metadata']]
                        )
                        imported_samples += 1
            else:
                # Old single-sample format
                for speaker_data in import_data["speakers"]:
                    # Convert to multi-sample format
                    sample_id = str(uuid.uuid4())
                    metadata = speaker_data["metadata"]
                    metadata.update({
                        "speaker_id": speaker_data["id"],
                        "speaker_name": metadata.get('name', 'Unknown'),
                        "sample_index": 1,
                        "sample_timestamp": metadata.get('last_seen', datetime.now().isoformat())
                    })
                    
                    self.collection.add(
                        ids=[sample_id],
                        embeddings=[speaker_data["embedding"]],
                        metadatas=[metadata]
                    )
                    
                    self.metadata["speakers"][speaker_data["id"]] = {
                        "name": metadata.get('name', 'Unknown'),
                        "imported_at": datetime.now().isoformat()
                    }
                    imported_speakers += 1
                    imported_samples += 1
            
            self.metadata["stats"]["total_speakers"] = len(self.metadata["speakers"])
            self.metadata["stats"]["total_embeddings"] = self.collection.count()
            self._save_metadata()
            
            print(f"Imported {imported_speakers} speakers ({imported_samples} samples) from {import_path}")
            return imported_speakers
            
        except Exception as e:
            print(f"Error importing database: {e}")
            return 0