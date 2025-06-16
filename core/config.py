# File: core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './results'))
    MODELS_DIR = Path(os.getenv('MODELS_DIR', './models'))
    SPEAKER_DB_DIR = Path(os.getenv('SPEAKER_DB_DIR', './speaker_database'))
    
    # API Keys
    OPENAI_API_KEY = os.getenv('API_KEY', os.getenv('OPENAI_API_KEY'))
    HF_TOKEN = os.getenv('HF_TOKEN')
    
    # Model Settings
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v2')
    WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cuda')
    WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'float16')
    WHISPER_BATCH_SIZE = int(os.getenv('WHISPER_BATCH_SIZE', '16'))
    
    # Processing Settings
    BATCH_SIZE_MINUTES = int(os.getenv('BATCH_SIZE_MINUTES', '40'))
    GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o')

    SPEAKER_MAX_SAMPLES = int(os.getenv('SPEAKER_MAX_SAMPLES', '20'))
    
    # Create directories
    @classmethod
    def setup_directories(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.SPEAKER_DB_DIR.mkdir(parents=True, exist_ok=True)