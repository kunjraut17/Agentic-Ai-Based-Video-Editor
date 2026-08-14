import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class SummarizerAgent:
    """
    The Summarizer Agent leverages Gemini's multimodal capabilities 
    to analyze video content and provide text-based insights.
    """
    
    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def get_insight(self, video_path, query, progress_callback=None):
        """
        Uploads the video to Gemini and gets a text-based response to the query.
        """
        if progress_callback: progress_callback(0.1, "☁️ Uploading video to AI...")
        
        print(f"Summarizer Agent: Uploading {video_path}...")
        video_file = genai.upload_file(path=video_path)
        print(f"Summarizer Agent: Processing video {video_file.name}...")
        if progress_callback: progress_callback(0.2, "🧠 AI is watching the video...")

        # Wait for the video to be processed by the API
        wait_count = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            wait_count += 1
            # Simulate slight progress while waiting (up to 80%)
            if progress_callback:
                simulated_progress = min(0.2 + (wait_count * 0.05), 0.8)
                progress_callback(simulated_progress, "🧠 AI is still analyzing...")
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")

        print("Summarizer Agent: Generating insights...")
        if progress_callback: progress_callback(0.9, "✍️ Formulating response...")
        
        system_prompt = (
            "You are the Summarizer Agent. Your task is to analyze the provided video content "
            "and provide a detailed, natural-sounding response to the user's query. "
            "Focus on visual details, actions, and the overall context of the video."
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    [video_file, f"{system_prompt}\n\nUser Question: {query}"],
                    request_options={"timeout": 600}
                )
                # Cleanup: Delete the file from Gemini's servers
                genai.delete_file(video_file.name)
                return response.text
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)
                    if progress_callback: progress_callback(0.9, f"⏳ Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if 'video_file' in locals():
                    genai.delete_file(video_file.name)
                raise e

if __name__ == "__main__":
    # Test logic
    # agent = SummarizerAgent()
    # print(agent.get_insight("sample.mp4", "What happens in this video?"))
    pass
