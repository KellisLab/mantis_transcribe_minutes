# File: core/utils.py
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename."""
    # Replace invalid characters
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    # Replace multiple spaces/underscores
    name = re.sub(r'[\s_]+', '_', name)
    # Limit length
    if len(name) > 100:
        name = name[:100]
    return name.strip('_')

def generate_output_path(input_path: str, input_type: str, base_output_dir: Path) -> Path:
    """Generate a unique output directory for the given input."""
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract a meaningful name
    if input_type in ['panopto_url', 'panopto_id']:
        # For Panopto, use ID as base
        base_name = f"panopto_{input_path[:8]}"
    else:
        # For files, use filename without extension
        base_name = Path(input_path).stem
    
    # Sanitize and create directory name
    safe_name = sanitize_filename(base_name)
    dir_name = f"{timestamp}_{safe_name}"
    
    # Create full path
    output_path = base_output_dir / dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path

def get_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def seconds_to_time_str(seconds):
    """
    Convert seconds to H:MM:SS format (e.g., 0:18:52)
    
    Args:
        seconds (int/float): Number of seconds
        
    Returns:
        str: Formatted time string
    """
    if pd.isna(seconds):
        return "00:00:00"
    
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"