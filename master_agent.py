import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class MasterAgent:
    """
    The Master Agent interprets natural language commands and decomposes them 
    into a sequence of actionable steps for specialized video editing agents.
    """
    
    def __init__(self, api_key=None, model_name="gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key not found. Please set GOOGLE_API_KEY in your environment or .env file.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def generate_plan(self, user_prompt: str) -> list:
        system_prompt = """
        You are the Master Agent of an Agentic AI Video Editing system. 
        Your job is to take a user's natural language request and decompose it into a sequence of operations 
        for specialized agents.

        Available Operations and Agents:
        1. ANALYSIS (Analysis Agent): Identify metadata, key moments, and visual content.
        2. TRIM (Editing Agent): Cut specific time segments. Use 'start_time' and 'end_time' (float seconds or MM:SS strings). Use 'end' for end of clip.
        3. HIGHLIGHT (Editing Agent): Select high-salience moments. Use 'target_duration' (seconds).
        4. SUMMARIZE (Editing Agent): Create a condensed version. Use 'target_duration' (seconds).
        5. RESIZE (Editing Agent): Change aspect ratio. Use 'aspect_ratio' (string, e.g., '9:16', '1:1', '16:9').
        6. ROTATE (Editing Agent): Rotate the video. Use 'angle' (number, e.g., 90, 180, 270).
        7. CAPTION (Captioning Agent): Generate and burn-in synchronized subtitles from audio.
        8. QUERY_INSIGHT (Summarizer Agent): Answer questions about the video content or provide a detailed description. Use 'query' (the user's question).
        9. STORYBOARD (Editing Agent): Sequence multiple clips into a narrative.
        10. STYLE_REFINEMENT (Quality Control Agent): Apply aesthetic filters or RL-based cinematic adjustments.

        Output Requirements:
        - You MUST output a valid JSON list of objects.
        - Each object must have: 'operation', 'parameters' (dict), and 'reasoning'.
        - Use consistent keys: 'start_time', 'end_time', 'target_duration'.

        Example Output for 'Trim the first 2 minutes':
        [
            {"operation": "TRIM", "parameters": {"start_time": 120, "end_time": "end"}, "reasoning": "Removing the first 2 minutes (120 seconds) as requested."}
        ]

        Example Output for 'Make it 9:16 for TikTok':
        [
            {"operation": "RESIZE", "parameters": {"aspect_ratio": "9:16"}, "reasoning": "Changing aspect ratio to 9:16 as requested for social media compatibility."}
        ]

        Example Output for 'Rotate the video 90 degrees clockwise':
        [
            {"operation": "ROTATE", "parameters": {"angle": 90}, "reasoning": "Rotating the video 90 degrees as requested."}
        ]

        Example Output for 'Add captions to this video':
        [
            {"operation": "CAPTION", "parameters": {}, "reasoning": "Generating and overlaying captions as requested."}
        ]

        Example Output for 'Make a 30s highlight':
        [
            {"operation": "ANALYSIS", "parameters": {"focus": "highlights"}, "reasoning": "Identify key segments."},
            {"operation": "HIGHLIGHT", "parameters": {"target_duration": 30}, "reasoning": "Extract the most significant segments to fit the 30s target."}
        ]

        Example Output for 'What is this video about?':
        [
            {"operation": "QUERY_INSIGHT", "parameters": {"query": "What is this video about?"}, "reasoning": "The user is asking for a general description of the video content."}
        ]

        Example Output for 'Does this video contain trekking? Describe it in detail':
        [
            {"operation": "QUERY_INSIGHT", "parameters": {"query": "Does this video contain trekking? Describe it in detail"}, "reasoning": "The user is asking for specific information about trekking in the video."}
        ]
        """
        
        prompt = f"{system_prompt}\n\nUser Request: '{user_prompt}'\n\nJSON Output:"
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
            )
        )
        
        try:
            plan = json.loads(response.text)
            return plan
        except json.JSONDecodeError:
            # Fallback for non-structured output if JSON mode fails
            print("Failed to parse JSON response. Raw output:")
            print(response.text)
            return []

if __name__ == "__main__":
    # Example usage
    try:
        agent = MasterAgent()
        sample_prompt = "I have a long recording of my vacation. Can you summarize it into the best 2 minutes and add some cinematic style?"
        print(f"User Prompt: {sample_prompt}")
        plan = agent.generate_plan(sample_prompt)
        print("\nGenerated Action Plan:")
        print(json.dumps(plan, indent=4))
    except Exception as e:
        print(f"Error: {e}")
