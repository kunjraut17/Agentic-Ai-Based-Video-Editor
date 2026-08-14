import streamlit as st
import os
import tempfile
import json
from master_agent import MasterAgent
from editing_agent import EditingAgent
from streamlit_logger import StreamlitMoviePyLogger
from summarizer_agent import SummarizerAgent
from captioning_agent import CaptioningAgent

# Page configuration
st.set_page_config(page_title="Agentic AI Video Editor", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #ff4b4b;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎬 Agentic AI-Based Smart Video Editor")
st.markdown("Automate your video editing with natural language. Powered by Multi-Agent AI.")
st.markdown("---")

# Sidebar Configuration
st.sidebar.title("⚙️ AI Configuration")

# User-provided specific free tier list
MODEL_MAPPING = {
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 2.0 Flash-Lite": "gemini-2.0-flash-lite-preview-02-05",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 3 Flash Preview": "gemini-3-flash-preview", # Future/Exp
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
    "Gemini 2.5 Flash-Lite Preview": "gemini-2.5-flash-lite-preview",
    "Gemini 2.5 Flash Native Audio": "gemini-2.5-flash-native-audio",
    "Gemini 2.5 Flash Preview TTS": "gemini-2.5-flash-preview-tts",
    "Gemma 3": "gemma-3-27b",
    "Gemma 3n": "gemma-3n-4b"
}

selected_model_label = st.sidebar.selectbox(
    "Select Gemini Model",
    list(MODEL_MAPPING.keys()),
    index=0
)
custom_model = st.sidebar.text_input("Or enter custom model ID:", placeholder="e.g., gemini-1.5-flash-latest")
selected_model = custom_model if custom_model else MODEL_MAPPING[selected_model_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("Agentic AI Video Editor v2.1\n\nNow with dynamic model selection and improved API robustness.")

# Initialize Agents
@st.cache_resource
def init_agents(model_name):
    return MasterAgent(model_name=model_name), EditingAgent(), SummarizerAgent(model_name=model_name), CaptioningAgent(model_name=model_name)

try:
    master_agent, editing_agent, summarizer_agent, captioning_agent = init_agents(selected_model)
except Exception as e:
    st.error(f"Failed to initialize agents: {e}")
    st.info("Check your .env file for the GOOGLE_API_KEY.")
    st.stop()

# File Upload Section
st.subheader("📁 Upload Video")
uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with st.expander("📺 Preview Uploaded Video", expanded=False):
        st.video(uploaded_file)
    
    st.markdown("---")
    
    # Tabs for different tasks
    tab_edit, tab_caption, tab_insight = st.tabs([
        "🎬 Edit & Highlight", 
        "💬 Smart Captions", 
        "🧠 Video AI Insight"
    ])

    # --- TAB 1: EDITING ---
    with tab_edit:
        st.subheader("Video Editing Agent")
        user_prompt = st.text_area(
            "What edits should I perform?", 
            placeholder="e.g., Create a 30 second highlight video and make it 9:16 vertical.",
            key="edit_prompt"
        )
        
        if st.button("🚀 Process Edits"):
            if not user_prompt:
                st.warning("Please enter an instruction.")
            else:
                progress_container = st.container()
                with progress_container:
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                
                with st.spinner(f"🤖 Analyzing edits with {selected_model}..."):
                    # Save to temp
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.getvalue())
                    tfile.close() 
                    input_path = tfile.name
                    
                    try:
                        plan = master_agent.generate_plan(user_prompt)
                        edit_steps = [s for s in plan if s['operation'] != 'QUERY_INSIGHT']
                        
                        if not edit_steps:
                            st.warning("No editing operations detected. Try being more specific.")
                        else:
                            custom_logger = StreamlitMoviePyLogger(progress_bar, status_text)
                            
                            for step in edit_steps:
                                if step['operation'] == 'CAPTION':
                                    def cb(p, t): progress_bar.progress(p); status_text.text(t)
                                    step['parameters']['captions'] = captioning_agent.transcribe(input_path, progress_callback=cb)
                            
                            status_text.text("🎞️ Executing video edits...")
                            output_path = editing_agent.execute_plan(input_path, edit_steps, user_prompt=user_prompt, logger=custom_logger)
                            
                            progress_container.empty()
                            st.balloons()
                            st.success("✨ Your edited video is ready!")
                            
                            if os.path.exists(output_path):
                                with open(output_path, 'rb') as vfile:
                                    video_bytes = vfile.read()
                                    st.video(video_bytes)
                                    st.download_button("📥 Download Edited Video", data=video_bytes, file_name="edited_video.mp4")
                        
                        with st.expander("🔍 AI Reasoning"): st.json(plan)
                    except Exception as e: st.error(f"Error: {e}")
                    finally: 
                        if os.path.exists(input_path): os.remove(input_path)

    # --- TAB 2: CAPTIONING ---
    with tab_caption:
        st.subheader("Captioning Agent")
        
        col1, col2 = st.columns(2)
        with col1:
            target_lang = st.selectbox(
                "Target Language", 
                ["English", "Hindi", "Spanish", "French", "German", "Japanese", "Chinese", "Korean", "Italian", "Portuguese", "Russian"]
            )
        with col2:
            st.info(f"Using **{selected_model}** for transcription. Accuracy is prioritized.")

        if st.button("✍️ Generate Captions"):
            progress_container = st.container()
            with progress_container:
                status_text = st.empty()
                progress_bar = st.progress(0)
            
            with st.spinner(f"🤖 Transcribing in {target_lang}..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.getvalue())
                tfile.close()
                input_path = tfile.name
                
                try:
                    def caption_cb(p, t): progress_bar.progress(p); status_text.text(t)
                    captions = captioning_agent.transcribe(input_path, language=target_lang, progress_callback=caption_cb)
                    
                    if not captions:
                        st.error("Could not generate captions. Check if your API quota is hit or if there is no audio.")
                    else:
                        status_text.text("🎞️ Burning captions into video...")
                        custom_logger = StreamlitMoviePyLogger(progress_bar, status_text)
                        
                        edit_plan = [{"operation": "CAPTION", "parameters": {"captions": captions}}]
                        output_path = editing_agent.execute_plan(input_path, edit_plan, logger=custom_logger)
                        
                        progress_container.empty()
                        st.balloons()
                        st.success(f"✨ Video captioned in {target_lang}!")
                        
                        if os.path.exists(output_path):
                            with open(output_path, 'rb') as vfile:
                                video_bytes = vfile.read()
                                st.video(video_bytes)
                                st.download_button("📥 Download Captioned Video", data=video_bytes, file_name="captioned_video.mp4")
                except Exception as e: st.error(f"Error: {e}")
                finally:
                    if os.path.exists(input_path): os.remove(input_path)

    # --- TAB 3: INSIGHT ---
    with tab_insight:
        st.subheader("Video AI Insight Agent")
        insight_prompt = st.text_area(
            "What would you like to know about this video?", 
            placeholder="e.g., Describe the main character's actions.",
            key="insight_prompt"
        )
        
        if st.button("🧐 Ask AI Agent"):
            if not insight_prompt:
                st.warning("Please enter a question.")
            else:
                progress_container = st.container()
                with progress_container:
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                
                with st.spinner("🤖 AI is watching..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.getvalue())
                    tfile.close()
                    input_path = tfile.name
                    
                    try:
                        def insight_cb(p, t): progress_bar.progress(p); status_text.text(t)
                        insight = summarizer_agent.get_insight(input_path, insight_prompt, progress_callback=insight_cb)
                        
                        progress_container.empty()
                        st.success("✅ Analysis Complete")
                        st.markdown(f"### 💡 AI Insight (via {selected_model})\n\n{insight}")
                    except Exception as e: st.error(f"Error: {e}")
                    finally:
                        if os.path.exists(input_path): os.remove(input_path)

else:
    st.info("Please upload a video file to begin.")
