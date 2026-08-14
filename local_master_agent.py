import ollama
import json

class LocalMasterAgent:
    """
    Local Master Agent using Ollama to interpret natural language.
    Decomposes requests into a sequence of video editing operations.
    """
    
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

    def generate_plan(self, user_prompt: str) -> list:
        system_prompt = """
        You are the Master Agent of a Video Editing system. 
        Decompose the user's request into a JSON list of objects.
        
        Operations:
        - TRIM: {start_time, end_time}
        - HIGHLIGHT: {target_duration}
        - SUMMARIZE: {target_duration}
        - RESIZE: {aspect_ratio} (e.g., '9:16', '1:1')
        - ROTATE: {angle} (90, 180)
        - CAPTION: {}
        
        Output Requirements:
        - Output ONLY a valid JSON list. No talk.
        - Example: [{"operation": "TRIM", "parameters": {"start_time": 0, "end_time": 10}, "reasoning": "..."}]
        """
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Request: {user_prompt}"},
            ])
            
            content = response['message']['content']
            # Find the JSON part if there is surrounding text
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx != -1 and end_idx != -1:
                return json.loads(content[start_idx:end_idx])
            return []
        except Exception as e:
            print(f"Ollama Error: {e}")
            return []

if __name__ == "__main__":
    # agent = LocalMasterAgent()
    # print(agent.generate_plan("trim the first 5 seconds"))
    pass
