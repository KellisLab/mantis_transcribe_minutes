# File: core/prompts.py
"""
Centralized prompt management with environment variable customization
"""
import os
from typing import Dict, Optional
from pathlib import Path
import json

class PromptManager:
    """Manages customizable GPT prompts with environment variable support."""
    
    # Default prompts
    DEFAULT_PROMPTS = {
        "SPEAKER_SUMMARY": '''Generate a concise summary of this speaker's contribution to a specific topic.

Instructions:
1. Return a JSON object with two fields: 'title' and 'content'
2. The 'title' should be a brief (3-7 words) descriptive title of the topic discussed
3. The 'content' should be a detailed summary of the speaker's contribution
4. MUST USE <b>bold</b> for important technical terms and concepts
5. Keep content to a single paragraph with no line breaks

TRANSCRIPT FROM {speaker} (TOPIC #{topic_num}):

{transcript}''',

        "BATCH_SUMMARY": '''Your task is to create a structured summary of this meeting section.

IMPORTANT: You MUST create summaries starting from the EARLIEST content in this batch, even if it seems introductory or less substantive

OUTPUT FORMAT REQUIREMENTS (CRITICAL):
1. Each topic must follow this EXACT format:
   **Topic Title - Speaker Name** (H:MM:SS): Content...
2. The format must be followed precisely with NO exceptions
3. Use only exact timestamps from the provided SPEAKER TIMESTAMPS section
4. BOLD important terms within the content: **terms**
5. Content should be in paragraph form (no bullet points or line breaks)

TIMESTAMP SELECTION RULES:
1. Choose the MOST RELEVANT timestamp from the provided options for each speaker
2. Match the timestamp to where the specific topic is actually discussed
3. NEVER create or modify timestamps - use only those provided

CONTENT REQUIREMENTS:
1. Thoroughly explain each topic with technical precision
2. Include interactions between different speakers
3. Be detailed and comprehensive
4. Do not hallucinate information
5. Do not include a concluding summary paragraph

{timestamp_reference}

MEETING TRANSCRIPT BATCH #{batch_number} ({start_time} - {end_time}):

{transcript}''',

        "SYSTEM_SPEAKER": "You are a technical meeting summarizer. MUST USE <b>bold</b> for important technical terms and concepts.",
        
        "SYSTEM_BATCH": "You are a technical meeting summarizer. NEVER modify the timestamps provided to you.",
        
        "TOPIC_EXTRACTION": '''Extract the main topics discussed in this transcript segment.

Requirements:
1. Return a JSON array of topics
2. Each topic should have: {"title": "Topic Name", "keywords": ["key1", "key2"], "importance": 1-5}
3. Focus on technical discussions and decisions
4. Maximum 5 topics per segment

Transcript:
{transcript}''',

        "ACTION_ITEMS": '''Extract action items and decisions from this meeting segment.

Requirements:
1. Return JSON: {"actions": [...], "decisions": [...]}
2. Each action: {"task": "...", "assignee": "...", "deadline": "..."}
3. Each decision: {"decision": "...", "rationale": "...", "stakeholders": [...]}
4. Only include explicitly stated items

Transcript:
{transcript}''',
        
        "SYSTEM_SCREENSHOT": """You are a helpful assistant that analyzes meeting transcripts and identifies the most important moments for screenshots.
        Focus on key moments that visually represent the main discussion points.
        You will be shown screenshots from a meeting along with their timestamps and context.
        Your task is to select the most representative screenshots that best capture the key moments of the discussion.
        For each selected screenshot, provide a brief caption for each screenshot.
        """,
        
        "SPEAKER_SCREENSHOT_CANDIDATES": """Analyze the following meeting transcript from {speaker} and identify the 5 most important timestamps given the provided time stamps 
        that could potentially best capture the key moments of the discussion, especially where there will be visual cues or demonstrations. 
        
        Topic: {topic_title}
        Summary: {topic_content}
        
        Transcript:
        {transcript}
        
        For each timestamp, provide a brief justification for why it's important.
        
        Return a JSON object with the following structure:
        {{
            "candidates": [
                {{
                    "timestamp": 123.45,
                    "justification": "Brief description of why this moment is important for a screenshot",
                }}
            ]
        }}
        """,
        
        "SCREENSHOT_SELECTION": """Analyze the following meeting topic and select the most relevant screenshots.
        
        **Topic:** {topic_title}
        **Summary:** {topic_content}
        
        **Instructions:**
        1. Review each screenshot and its timestamp
        2. Select up to three screenshots that best represent the key moments of this discussion (if there are none that are key moments or only show the speakers without other visual cues, return an empty array)
        3. Consider visual clarity, relevance to the topic, and importance of the moment
        4. Return your selection as a JSON object with the selected indices and reasoning
        5. Given the selected indices and corresponding images, provide for each corresponding index 
         an accurate and a meaningful description, which will be used as a caption  (This must be very high quality and very professional)
        6. Each description of the image should utilize both the image and the context provided in the summary to generate a meaningful caption 
        
       Return a JSON object with the following structure:
       {{
            "selected_indices": [number, ...],  // Array of 1-3 integers representing the indices of selected screenshots
            "reasoning": "string",              // Explanation for why these screenshots were selected
            "caption": ["string", ...]             // Array of captions that corresponds to each selected screenshots from left to right in the array
       }}


       Example Response: 
       {{
            "selected_indices": [0, 2],
            "reasoning": "Screenshot 0 shows the main discussion point clearly, while screenshot 2 captures an important visual demonstration."
            "caption": ["A visual demonstration of the Mantis embedding space.", "A visual demonstration of the Mantis embedding saoce."]
        }}
        """
    }
    
    def __init__(self, custom_prompts_file: Optional[Path] = None):
        """
        Initialize prompt manager.
        
        Args:
            custom_prompts_file: Path to JSON file with custom prompts
        """
        self.prompts = self.DEFAULT_PROMPTS.copy()
        self.custom_prompts_file = custom_prompts_file
        
        # Load custom prompts from file if provided
        if custom_prompts_file and custom_prompts_file.exists():
            self._load_custom_prompts(custom_prompts_file)
        
        # Override with environment variables
        self._load_env_prompts()
    
    def _load_custom_prompts(self, file_path: Path):
        """Load custom prompts from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_prompts = json.load(f)
                self.prompts.update(custom_prompts)
                print(f"Loaded {len(custom_prompts)} custom prompts from {file_path}")
        except Exception as e:
            print(f"Warning: Failed to load custom prompts: {e}")
    
    def _load_env_prompts(self):
        """Load prompts from environment variables."""
        # Check for individual prompt overrides
        for prompt_key in self.prompts.keys():
            env_key = f"GPT_PROMPT_{prompt_key}"
            if env_value := os.getenv(env_key):
                self.prompts[prompt_key] = env_value
                print(f"Loaded {prompt_key} from environment variable {env_key}")
        
        # Check for prompts file path in env
        if prompts_file_env := os.getenv("GPT_PROMPTS_FILE"):
            prompts_path = Path(prompts_file_env)
            if prompts_path.exists():
                self._load_custom_prompts(prompts_path)
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt by name with variable substitution.
        
        Args:
            prompt_name: Name of the prompt
            **kwargs: Variables to substitute in the prompt
            
        Returns:
            Formatted prompt string
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        prompt = self.prompts[prompt_name]
        
        # Substitute variables
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable for prompt {prompt_name}: {e}")
    
    def list_prompts(self) -> Dict[str, str]:
        """List all available prompts."""
        return self.prompts.copy()
    
    def save_prompts(self, file_path: Path):
        """Save current prompts to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.prompts, f, indent=2)
        print(f"Saved prompts to {file_path}")