# File: core/input_detector.py
import re
import os
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Optional

class InputType(Enum):
    PANOPTO_URL = "panopto_url"
    PANOPTO_ID = "panopto_id"
    MEDIA_FILE = "media_file"
    SUBTITLE_FILE = "subtitle_file"
    EXCEL_FILE = "excel_file"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"

class InputDetector:
    # Supported file extensions
    MEDIA_EXTENSIONS = {'.mp4', '.mp3', '.wav', '.m4a', '.avi', '.mov', '.flac', '.ogg', '.webm'}
    SUBTITLE_EXTENSIONS = {'.vtt', '.srt', '.sbv', '.sub'}
    EXCEL_EXTENSIONS = {'.xlsx', '.xls'}
    
    # Panopto patterns
    PANOPTO_URL_PATTERN = re.compile(r'panopto\.com/Panopto/Pages/Viewer\.aspx\?id=([a-f0-9\-]+)', re.IGNORECASE)
    PANOPTO_ID_PATTERN = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', re.IGNORECASE)
    
    @classmethod
    def detect(cls, input_path: str) -> Tuple[InputType, Optional[str]]:
        """
        Detect the type of input and return the type with extracted metadata.
        
        Returns:
            Tuple of (InputType, metadata) where metadata could be:
            - For Panopto URL/ID: the video ID
            - For files: the absolute path
            - For directory: the absolute path
        """
        input_path = input_path.strip()
        
        # Check if it's a URL
        if cls._is_url(input_path):
            # Check for Panopto URL
            match = cls.PANOPTO_URL_PATTERN.search(input_path)
            if match:
                return InputType.PANOPTO_URL, match.group(1)
            return InputType.UNKNOWN, None
        
        # Check if it's a Panopto ID
        if cls.PANOPTO_ID_PATTERN.match(input_path):
            return InputType.PANOPTO_ID, input_path
        
        # Check if it's a file or directory
        path = Path(input_path)
        
        if path.is_dir():
            return InputType.DIRECTORY, str(path.absolute())
        
        if path.is_file():
            ext = path.suffix.lower()
            
            if ext in cls.MEDIA_EXTENSIONS:
                return InputType.MEDIA_FILE, str(path.absolute())
            elif ext in cls.SUBTITLE_EXTENSIONS:
                return InputType.SUBTITLE_FILE, str(path.absolute())
            elif ext in cls.EXCEL_EXTENSIONS:
                return InputType.EXCEL_FILE, str(path.absolute())
        
        return InputType.UNKNOWN, None
    
    @staticmethod
    def _is_url(string: str) -> bool:
        """Check if string is a URL."""
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except:
            return False