import whisper
import os
import torch

class LocalCaptioningAgent:
    """
    Local Captioning Agent using OpenAI's Whisper model.
    Runs entirely on the local CPU or GPU.
    """
    
    def __init__(self, model_size="base"):
        # Select device: GPU (cuda) if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, video_path, language=None, progress_callback=None):
        """
        Transcribes audio from a video file locally.
        """
        if progress_callback: progress_callback(0.3, "🧠 Whisper is analyzing audio locally...")
        
        # Whisper can take the video path directly as it extracts audio internally
        options = {}
        if language and language.lower() != "english":
            options["language"] = language.lower()

        result = self.model.transcribe(video_path, **options)
        
        captions = []
        for segment in result['segments']:
            captions.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })

        if progress_callback: progress_callback(1.0, "✅ Local Transcription complete!")
        return captions

if __name__ == "__main__":
    # Example usage:
    # agent = LocalCaptioningAgent()
    # print(agent.transcribe("video.mp4"))
    pass
