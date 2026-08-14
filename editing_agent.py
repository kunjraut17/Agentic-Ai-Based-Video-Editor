import os
import warnings
# Suppress harmless MoviePy warnings about reading last frames
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")
from moviepy import VideoFileClip, concatenate_videoclips, vfx, TextClip, CompositeVideoClip
import numpy as np

class EditingAgent:
    """
    The Editing Agent executes the physical video manipulations 
    defined in the action plan from the Master Agent.
    """
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def _get_system_font(self):
        """Helper to find a valid system font for Windows with Hindi support."""
        system_fonts = [
            r"C:\Windows\Fonts\mangal.ttf",      # Preferred for Hindi
            r"C:\Windows\Fonts\Nirmala.ttc",    # Modern Indic font
            r"C:\Windows\Fonts\arial.ttf",      # Basic fallback
            r"C:\Windows\Fonts\SegoeUI.ttf",
            r"C:\Windows\Fonts\Calibri.ttf"
        ]
        for font in system_fonts:
            if os.path.exists(font):
                return font
        return None # Falls back to Pillow default

    def execute_plan(self, input_video_path, plan, user_prompt="", logger=None):
        """
        Takes an input video and a list of operations to perform.
        Returns the path to the final edited video.
        """
        base_clip = VideoFileClip(input_video_path)
        clips_to_close = [base_clip]
        current_clip = base_clip
        final_video_path = os.path.join(self.output_dir, "final_edit.mp4")
        
        try:
            for step in plan:
                operation = step.get("operation")
                params = step.get("parameters", {})
                
                print(f"Executing {operation} with params {params}...")
                
                if operation == "TRIM":
                    # Handle multiple possible parameter names
                    start = params.get("start_time") or params.get("start_offset") or 0
                    end = params.get("end_time") or params.get("end_point") or current_clip.duration
                    
                    def parse_time(val, default):
                        if val is None: return default
                        if isinstance(val, (int, float)): return val
                        if isinstance(val, str):
                            val = val.lower().strip()
                            if val in ["end", "eof", "max"]: return current_clip.duration
                            if ":" in val:
                                try:
                                    parts = list(map(float, val.split(":")))
                                    if len(parts) == 2: return parts[0] * 60 + parts[1]
                                    if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
                                except: return default
                            try: return float(val)
                            except: return default
                        return default

                    start_sec = parse_time(start, 0)
                    end_sec = parse_time(end, current_clip.duration)
                    
                    # Safety checks
                    start_sec = max(0, min(start_sec, current_clip.duration))
                    end_sec = max(start_sec, min(end_sec, current_clip.duration))
                    
                    if end_sec > start_sec:
                        current_clip = current_clip.subclipped(start_sec, end_sec)
                        clips_to_close.append(current_clip)
                    
                elif operation == "SUMMARIZE" or operation == "HIGHLIGHT":
                    duration = current_clip.duration
                    # Check both 'target_duration' and 'duration' for flexibility
                    target_duration = params.get("target_duration") or params.get("duration") or (duration / 2)
                    
                    if isinstance(target_duration, str):
                        try:
                            target_duration = float(target_duration)
                        except:
                            target_duration = duration / 2

                    if duration > target_duration:
                        # Improved logic: Select N segments throughout the video to meet target_duration
                        # We'll take 5 segments of equal length
                        num_segments = 5
                        segment_length = target_duration / num_segments
                        segments = []
                        
                        for i in range(num_segments):
                            start_p = (i / num_segments) * (duration - segment_length)
                            segments.append(current_clip.subclipped(start_p, start_p + segment_length))
                        
                        current_clip = concatenate_videoclips(segments)
                        clips_to_close.append(current_clip)
                        # Concatenate already includes children, but we'll track it
                
                elif operation == "RESIZE":
                    ratio_str = params.get("aspect_ratio", "16:9")
                    w, h = current_clip.size
                    
                    target_ratio = None
                    if ratio_str == "9:16": target_ratio = 9/16
                    elif ratio_str == "1:1": target_ratio = 1.0
                    elif ratio_str == "16:9": target_ratio = 16/9
                    elif ":" in ratio_str:
                        try:
                            rw, rh = map(float, ratio_str.split(":"))
                            target_ratio = rw / rh
                        except: pass
                    
                    if target_ratio:
                        current_ratio = w / h
                        if abs(current_ratio - target_ratio) > 0.01:
                            # We'll use crop and center logic
                            if current_ratio > target_ratio:
                                # Video is too wide, crop sides
                                new_w = h * target_ratio
                                x1 = (w - new_w) / 2
                                current_clip = current_clip.cropped(x1=x1, y1=0, width=new_w, height=h)
                            else:
                                # Video is too tall, crop top/bottom
                                new_h = w / target_ratio
                                y1 = (h - new_h) / 2
                                current_clip = current_clip.cropped(x1=0, y1=y1, width=w, height=new_h)
                            
                            clips_to_close.append(current_clip)
                            
                elif operation == "ROTATE":
                    angle = params.get("angle", 0)
                    try:
                        angle = float(angle)
                        if angle != 0:
                            current_clip = current_clip.rotated(angle)
                            clips_to_close.append(current_clip)
                    except:
                        pass

                elif operation == "CAPTION":
                    captions = params.get("captions", [])
                    if captions:
                        caption_clips = []
                        w, h = current_clip.size
                        for cap in captions:
                            try:
                                start = float(cap.get('start', 0))
                                end = float(cap.get('end', 0))
                                text = cap.get('text', "")
                                
                                # Cap start/end to clip duration
                                start = min(start, current_clip.duration)
                                end = min(end, current_clip.duration)
                                
                                if start >= end: continue

                                # Create individual text clip
                                txt_clip = TextClip(
                                    text=text,
                                    font=self._get_system_font(),
                                    font_size=int(h * 0.06),
                                    color="white",
                                    stroke_color="black",
                                    stroke_width=2,
                                    size=(int(w * 0.85), None),
                                    method="caption"
                                ).with_start(start).with_duration(end - start).with_position(("center", h * 0.85))
                                
                                caption_clips.append(txt_clip)
                                clips_to_close.append(txt_clip)
                            except Exception as e:
                                print(f"Error creating caption clip: {e}")
                                continue
                        
                        if caption_clips:
                            # Overlay all caption clips onto the current video
                            current_clip = CompositeVideoClip([current_clip] + caption_clips)
                            clips_to_close.append(current_clip)

                elif operation == "STORYBOARD":
                    # For now, storyboard just ensures we have a valid sequence
                    # In a complex app, this would handle re-ordering multiple clips
                    pass
                
                elif operation == "STYLE_REFINEMENT":
                    style = (params.get("style", "") or params.get("preset", "") or "").lower()
                    vibrancy = (params.get("vibrancy", "") or params.get("aesthetic_target", "") or "").lower()
                    
                    effects = []
                    
                    # 1. Cinematic look: Higher contrast, slightly darker mids
                    if "cinematic" in style or "cinematic" in user_prompt.lower():
                        # MoviePy 2.x uses with_effects
                        effects.append(vfx.LumContrast(lum=0, contrast=0.2)) # Boost contrast
                    
                    # 2. Vibrant look: Saturation boost
                    if "vibrant" in vibrancy or "vibrant" in user_prompt.lower() or "high" in vibrancy:
                        # Define a saturator function
                        def increase_saturation(frame):
                            # Convert to float to avoid overflow
                            frame_f = frame.astype(float)
                            # Simple saturation logic: pull colors away from the mean luminance
                            avg = frame_f.mean(axis=2, keepdims=True)
                            saturated = avg + (frame_f - avg) * 1.3 # 1.3x saturation
                            return np.clip(saturated, 0, 255).astype(np.uint8)
                        
                        current_clip = current_clip.image_transform(increase_saturation)
                        clips_to_close.append(current_clip)

                    if effects:
                        current_clip = current_clip.with_effects(effects)
                        clips_to_close.append(current_clip)

            # Finalize and write output with optional logger
            current_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac", logger=logger)
            return final_video_path
            
        finally:
            for clip in clips_to_close:
                try:
                    clip.close()
                except:
                    pass

if __name__ == "__main__":
    # Test logic
    # agent = EditingAgent()
    # agent.execute_plan("sample.mp4", [{"operation": "TRIM", "parameters": {"start_time": 0, "end_time": 5}}])
    pass
