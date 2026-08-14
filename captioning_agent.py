import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class CaptioningAgent:
    """
    The Captioning Agent uses Gemini's multimodal capabilities to transcribe
    audio from a video and generate synchronized timestamps for subtitling.
    """
    
    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def transcribe(self, video_path, language="English", progress_callback=None):
        """
        Transcribes the video and returns a list of caption dictionaries.
        Each dictionary contains 'start', 'end', and 'text'.
        """
        if progress_callback: progress_callback(0.1, f"☁️ Uploading for {language} transcription...")
        
        video_file = genai.upload_file(path=video_path)
        
        while video_file.state.name == "PROCESSING":
            if progress_callback: progress_callback(0.3, "🧠 AI is listening to the video...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Transcription failed: {video_file.state.name}")

        if progress_callback: progress_callback(0.7, f"✍️ Generating synchronized {language} captions...")
        
        system_prompt = (
            f"You are a professional video transcriber and subtitler fluent in {language}. "
            f"Your mission is to transcribe EVERY spoken word in this video with 100% verbatim accuracy into {language}. "
            "DO NOT skip background talks, DO NOT summarize, and DO NOT paraphrase. "
            "For each segment, provide: "
            "1. 'start': Precise start time in seconds (float). "
            "2. 'end': Precise end time in seconds (float). "
            "3. 'text': The exact verbatim transcript. "
            "Ensure the timestamps are perfectly synchronized with the audio onset and offset. "
            "Return ONLY a clean JSON list of objects. "
            "Format: [{\"start\": 0.0, \"end\": 2.5, \"text\": \"Verbatim text\"}, ...]"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    [video_file, system_prompt],
                    generation_config={"response_mime_type": "application/json"}
                )
                # Cleanup
                genai.delete_file(video_file.name)
                
                captions = json.loads(response.text)
                if progress_callback: progress_callback(1.0, "✅ Transcription complete!")
                return captions
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)
                    if progress_callback: progress_callback(0.7, f"⏳ Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                print(f"Error parsing Gemini response or Rate Limit: {e}")
                if 'video_file' in locals():
                    genai.delete_file(video_file.name)
                return []

if __name__ == "__main__":
    # Test logic
    # agent = CaptioningAgent()
    # print(agent.transcribe("sample.mp4"))
    pass
