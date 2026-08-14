from proglog import ProgressBarLogger
import streamlit as st

class StreamlitMoviePyLogger(ProgressBarLogger):
    """
    A custom logger that hooks into MoviePy's progress reporting 
    and updates a Streamlit progress bar.
    """
    def __init__(self, progress_bar, status_text):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.last_progress = -1

    def callback(self, **changes):
        # MoviePy passes status updates through this callback
        bars = self.state.get('bars')
        if not bars:
            return
            
        # MoviePy 2.x usually uses 'frame_index'. Earlier versions or audio use 'chunk'.
        # We prioritize 'frame_index' as it represents the bulk of the work.
        target_bar = bars.get('frame_index') or bars.get('chunk')
        
        # If neither found, pick the first available bar with a total
        if not target_bar:
            for name, bar in bars.items():
                if bar.get('total'):
                    target_bar = bar
                    break
        
        if target_bar and target_bar.get('total') and target_bar.get('total') > 0:
            progress = target_bar['index'] / target_bar['total']
            # Clamp progress
            progress = max(0.0, min(1.0, progress))
            percentage = int(progress * 100)
            
            # Update Streamlit components
            if percentage != self.last_progress:
                self.progress_bar.progress(progress)
                self.status_text.text(f"Generation Progress: {percentage}%")
                self.last_progress = percentage
