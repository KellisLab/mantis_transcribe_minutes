# File: speaker_id/__init__.py
"""
Speaker Identification Module

Provides voice fingerprinting and speaker recognition capabilities
for the transcript processing platform.
"""

from .fingerprint_manager import SpeakerFingerprintManager

__version__ = "1.0.0"
__all__ = ["SpeakerFingerprintManager"]
# Speaker recognition imports
from .fingerprint_manager import SpeakerFingerprintManager
