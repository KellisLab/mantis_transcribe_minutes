# File: processors/panopto_processor.py
import os
import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup

from processors.base_processor import BaseProcessor
from core.utils import sanitize_filename

class PanoptoProcessor(BaseProcessor):
    """Processor for handling Panopto video URLs and IDs."""
    
    # Map simple language codes to Panopto's expected format
    LANGUAGE_MAP = {
        'en': 'English_USA',
        'es': 'Spanish',
        # Add other language mappings as needed
    }
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.base_url = "https://mit.hosted.panopto.com"
    
    def process(self, input_path: str, video_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a Panopto URL or ID to download transcript.
        
        Args:
            input_path: Panopto URL or ID
            video_id: Pre-extracted video ID (optional)
            **kwargs: Additional options (language, meeting_name, etc.)
        
        Returns:
            Dict with processing results
        """
        # Extract video ID if not provided
        if not video_id:
            video_id = self._extract_video_id(input_path)
            if not video_id:
                return {
                    'success': False,
                    'error': 'Could not extract valid Panopto video ID'
                }
        
        # Extract metadata
        metadata = self._extract_metadata(video_id)
        
        # Use provided meeting name or extracted one
        meeting_name = kwargs.get('meeting_name', metadata.get('meeting_name', f'Panopto_{video_id[:8]}'))
        metadata['meeting_name'] = meeting_name
        
        # Download transcript
        language_code = kwargs.get('language', 'en')
        language = self.LANGUAGE_MAP.get(language_code, language_code)
        transcript_file = self._download_transcript(video_id, language, meeting_name)
        
        if not transcript_file:
            return {
                'success': False,
                'error': 'Failed to download transcript',
                'metadata': metadata
            }
        
        # Save metadata
        metadata['video_id'] = video_id
        metadata['source_url'] = f"{self.base_url}/Panopto/Pages/Viewer.aspx?id={video_id}"
        metadata['transcript_file'] = str(transcript_file)
        metadata['language'] = language
        
        self.save_metadata(metadata)
        
        return {
            'success': True,
            'transcript_file': transcript_file,
            'metadata': metadata,
            'video_id': video_id
        }
    
    def _extract_video_id(self, input_path: str) -> Optional[str]:
        """Extract video ID from URL or return as-is if already an ID."""
        # Check if it's already a video ID
        id_pattern = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', re.IGNORECASE)
        if id_pattern.match(input_path):
            return input_path
        
        # Extract from URL
        url_pattern = re.compile(r'id=([a-f0-9\-]+)', re.IGNORECASE)
        match = url_pattern.search(input_path)
        
        return match.group(1) if match else None
    
    def _extract_metadata(self, video_id: str) -> Dict[str, Any]:
        """Extract meeting name and other metadata from Panopto page."""
        metadata = {
            'meeting_name': f'Panopto_{video_id[:8]}',  # Default name
            'extracted': False
        }
        
        try:
            # Try to get the viewer page
            url = f"{self.base_url}/Panopto/Pages/Viewer.aspx?id={video_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try different methods to extract title
                # Method 1: Page title
                if soup.title and soup.title.string:
                    title = soup.title.string
                    if " - Panopto" in title:
                        metadata['meeting_name'] = title.split(" - Panopto")[0].strip()
                        metadata['extracted'] = True
                
                # Method 2: Meta tags
                if not metadata['extracted']:
                    meta_title = soup.find('meta', property='og:title')
                    if meta_title and meta_title.get('content'):
                        metadata['meeting_name'] = meta_title.get('content').strip()
                        metadata['extracted'] = True
                
                print(f"Extracted meeting name: {metadata['meeting_name']}")
            
        except Exception as e:
            print(f"Warning: Could not extract metadata: {e}")
        
        return metadata
    
    def _download_transcript(self, video_id: str, language: str = "English_USA", meeting_name: str = None) -> Optional[Path]:
        """Download SRT transcript from Panopto."""
        # Construct the transcript URL
        transcript_url = f"{self.base_url}/Panopto/Pages/Transcription/GenerateSRT.ashx?id={video_id}&language={language}"
        
        try:
            print(f"Downloading transcript from Panopto...")
            response = requests.get(transcript_url, timeout=30)
            
            if response.status_code == 200 and response.content:
                # Check if we got actual SRT content
                content_start = response.content[:100].decode('utf-8', errors='ignore')
                if not content_start.strip() or '<html' in content_start.lower() or 'login' in content_start.lower():
                    print("Warning: Received HTML/login page instead of SRT content.")
                    return None
                
                # Save the transcript with proper meeting name
                if meeting_name:
                    safe_name = sanitize_filename(meeting_name)
                else:
                    safe_name = sanitize_filename(self.metadata.get('meeting_name', f'panopto_{video_id[:8]}'))
                
                srt_file = self.output_dir / f"{safe_name}.srt"
                
                with open(srt_file, 'wb') as f:
                    f.write(response.content)
                
                print(f"Transcript saved to: {srt_file}")
                return srt_file
            else:
                print(f"Failed to download transcript. Status code: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error downloading transcript: {e}")
            return None

    def capture_screenshot(
        self, 
        video_id: str, 
        timestamp_seconds: int, 
        # Default crop to capture the main video area (black region)
        # Format: (width, height, x_offset, y_offset)
        crop: tuple = (1440, 905, 670, 0),  # Adjusted to better fit the black region
        timeout_sec: int = 40,
        max_retries: int = 3,
        viewport_size: dict = {"width": 1920, "height": 1080},
    ) -> Optional[bytes]:
        """
        Capture a screenshot from a Panopto video at the specified timestamp using a headless browser.
        Includes cropping functionality to only capture the video area
        
        Args:
            video_id: Panopto video ID
            timestamp_seconds: Timestamp in seconds
            crop: Tuple of (width, height, x_offset, y_offset) for the crop filter.
                If width or height is 0, the full width/height will be used.
                Example: (1280, 720, 100, 50) will crop to 1280x720 starting at (100, 50)
            timeout_sec: Timeout in seconds for browser operations
            max_retries: Maximum number of retry attempts
            viewport_size: Browser viewport size

        Returns:
            bytes: PNG image data or None if capture fails
        """
        from playwright.sync_api import sync_playwright
        from PIL import Image
        import io
        import time

        url = (
            self.base_url + "/Panopto/Pages/Viewer.aspx"
            f"?id={video_id}&start={timestamp_seconds}"
        )

        for attempt in range(1, max_retries + 1):
            try:
                with sync_playwright() as pw:
                    # Launch browser
                    browser = pw.chromium.launch(
                        channel="chrome",
                        headless=True,
                        args=["--no-sandbox", "--disable-gpu", "--disable-blink-features=AutomationControlled"]
                    )

                    # Create context and page with specified viewport
                    context = browser.new_context(viewport=viewport_size)
                    page = context.new_page()
                    page.set_default_timeout(timeout_sec * 1000)
                    
                    # Hide WebDriver
                    page.add_init_script(
                        "Object.defineProperty(navigator, 'webdriver', {get:() => undefined});"
                    )

                    # Navigate to video
                    page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_load_state("networkidle")

                    # Wait for video to be ready
                    page.wait_for_function("""
                    () => {
                        const v = document.getElementById('secondaryVideo');
                        return v && v.readyState >= 3;
                    }
                    """, timeout=timeout_sec * 1000)

                    # Click play button
                    page.wait_for_selector('#playButton', timeout=timeout_sec * 1000)
                    page.click('#playButton')

                    # Wait for video elements
                    page.wait_for_selector("video#primaryVideo", state="attached", timeout=timeout_sec * 1000)
                    page.wait_for_selector("video#secondaryVideo", state="attached", timeout=timeout_sec * 1000)

                    # Seek to exact timestamp
                    page.wait_for_function("""
                        async (targetTime) => {
                            const video = document.getElementById('secondaryVideo');
                            if (!video) return false;
                            
                            if (video.readyState < 3) return false;
                            
                            video.currentTime = targetTime;
                            await new Promise(r => setTimeout(r, 100));
                            video.play();
                            video.pause();
                            return Math.abs(video.currentTime - targetTime) < 0.1;
                        }
                    """, arg=timestamp_seconds, timeout=timeout_sec * 1000)

                    # Take screenshot
                    png_bytes = page.screenshot(full_page=True)
                    browser.close()

                    # Process image with crop and resize
                    with Image.open(io.BytesIO(png_bytes)) as img:
                        # Apply crop if specified
                        crop_w, crop_h, crop_x, crop_y = crop
                        if crop_w > 0 and crop_h > 0:
                            # Ensure crop dimensions are within image bounds
                            img_width, img_height = img.size
                            right = min(crop_x + crop_w, img_width)
                            bottom = min(crop_y + crop_h, img_height)
                            left = min(crop_x, img_width - 1)
                            top = min(crop_y, img_height - 1)
                            
                            # Only crop if the crop area is valid
                            if right > left and bottom > top:
                                img = img.crop((left, top, right, bottom))
                        
                        # Convert back to bytes
                        output = io.BytesIO()
                        img.save(output, format='PNG')
                        return output.getvalue()

            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    return None
                time.sleep(2 ** attempt)
        
        return None