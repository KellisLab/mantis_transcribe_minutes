# File: processors/base_processor.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

class BaseProcessor(ABC):
    """Base class for all processors."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metadata = {}
    
    @abstractmethod
    def process(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process the input and return results.
        
        Returns:
            Dict containing:
            - 'success': bool
            - 'transcript_file': Path to transcript file (if applicable)
            - 'metadata': Dict of metadata
            - 'error': Optional error message
        """
        pass
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = "metadata.json"):
        """Save metadata to JSON file."""
        import json
        metadata_path = self.output_dir / filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        return metadata_path