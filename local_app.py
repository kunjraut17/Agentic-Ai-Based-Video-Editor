import streamlit as st
import os
import tempfile
from local_master_agent import LocalMasterAgent
from local_captioning_agent import LocalCaptioningAgent
from editing_agent import EditingAgent # Reusable
from streamlit_logger import StreamlitMoviePyLogger

st.set_page_config(page_title="LOCAL AI Video Editor", page_icon="🏠", layout="wide")

st.title("🏠 Standalone Local AI Video Editor")
st.markdown("100% Personal & Private. No Cloud Quotas. (Uses Whisper & Ollama)")
st.markdown("---")

# Initialization
@st.cache_resource
def init_local_agents():
    # Attempt to load agents
    try:
        return LocalMasterAgent(), LocalCaptioningAgent(), EditingAgent()
    except Exception as e:
        st.error(f"Initialization Failed: {e}")
        st.info("Ensure you have installed: pip install openai-whisper ollama torch")
        st.stop()

master, captioner, editor = init_local_agents()

uploaded_file = st.file_uploader("Upload Local Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.video(uploaded_file)
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🎬 Edit", "💬 Caption"])
    
    with tab1:
        prompt = st.text_area("What should I do?")
        if st.button("🚀 Process Locally"):
            with st.spinner("Analyzing with local LLM..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.getvalue())
                input_path = tfile.name
                tfile.close()

                try:
                    plan = master.generate_plan(prompt)
                    st.json(plan)
                    
                    status = st.empty()
                    bar = st.progress(0)
                    logger = StreamlitMoviePyLogger(bar, status)
                    
                    output_path = editor.execute_plan(input_path, plan, logger=logger)
                    
                    st.success("Done!")
                    st.video(output_path)
                finally:
                    if os.path.exists(input_path): os.remove(input_path)

    with tab2:
        lang = st.selectbox("Language", ["English", "Hindi", "Spanish", "French"])
        if st.button("✍️ Local Transcribe"):
            with st.spinner("Whisper is transcribing (this may take a few minutes)..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.getvalue())
                input_path = tfile.name
                tfile.close()
                
                try:
                    def cb(p, t): st.text(t)
                    captions = captioner.transcribe(input_path, language=lang, progress_callback=cb)
                    
                    # Burn in
                    plan = [{"operation": "CAPTION", "parameters": {"captions": captions}}]
                    output_path = editor.execute_plan(input_path, plan)
                    st.video(output_path)
                finally:
                    if os.path.exists(input_path): os.remove(input_path)
else:
    st.info("Upload a video to start the local engine.")
