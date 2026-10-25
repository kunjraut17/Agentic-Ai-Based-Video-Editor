# 🎬 Agentic AI-Based Smart Video Editor

> A multi-agent AI video editing system that converts natural-language instructions into executable video-editing workflows.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![MoviePy](https://img.shields.io/badge/Video-MoviePy-orange)](https://zulko.github.io/moviepy/)
[![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4?logo=google)](https://ai.google.dev/)

## 📌 Overview

The **Agentic AI-Based Smart Video Editor** is an AI-powered video manipulation platform designed to make video editing accessible through natural language.

Instead of manually selecting timestamps, cropping frames, generating subtitles, and answering questions about a video, users can describe what they want:

> "Create a 30-second highlight video and make it vertical for social media."

A **Master Agent** interprets the request, decomposes it into structured operations, and delegates the work to specialized agents and the video-editing engine.

### Core idea

```text
Natural-language request
          │
          ▼
   ┌───────────────┐
   │  Master Agent │
   │ Gemini LLM    │
   └───────┬───────┘
           │
      Structured Plan
           │
    ┌──────┼───────────────┐
    ▼      ▼               ▼
Editing  Captioning   Video Insight
 Agent     Agent          Agent
    │        │               │
    └────────┴───────┬───────┘
                     ▼
              MoviePy Engine
                     │
                     ▼
             Edited / Captioned
                  Video
```

## ✨ Features

### 🎬 Natural-language video editing
- Trim videos using timestamps or natural-language instructions
- Generate highlight/summary clips
- Resize videos to common aspect ratios such as `9:16`, `1:1`, and `16:9`
- Rotate videos
- Apply basic cinematic and vibrant visual refinements
- Execute multiple editing operations as one agent-generated plan

### 💬 AI-powered captions
- Multimodal video transcription with Gemini
- Timestamped caption generation
- Multi-language caption support
- Burn captions directly into the output video

### 🧠 Video understanding
Ask questions about an uploaded video and receive an AI-generated response based on its visual and contextual content.

### 🏠 Local AI mode
The repository also contains an experimental local pipeline using:
- Ollama for local LLM-based planning
- OpenAI Whisper for local transcription
- MoviePy for local video processing

## 🧩 Multi-Agent Architecture

| Agent / Component | Responsibility |
|---|---|
| **Master Agent** | Converts natural-language requests into structured JSON editing plans |
| **Editing Agent** | Executes video operations with MoviePy |
| **Captioning Agent** | Generates synchronized multilingual captions |
| **Summarizer Agent** | Answers questions about video content |
| **Streamlit UI** | Provides the interactive application |
| **Local Master Agent** | Optional offline/local planning through Ollama |
| **Local Captioning Agent** | Optional local transcription through Whisper |

### Supported operation types

- `TRIM`
- `HIGHLIGHT`
- `SUMMARIZE`
- `RESIZE`
- `ROTATE`
- `CAPTION`
- `QUERY_INSIGHT`
- `STORYBOARD`
- `STYLE_REFINEMENT`

## 🛠️ Tech Stack

**AI / LLM**
- Google Gemini
- Ollama + Llama 3 (optional local mode)
- OpenAI Whisper (optional local mode)

**Video / Media**
- MoviePy
- FFmpeg
- NumPy

**Application**
- Python
- Streamlit
- Docker

**Deployment**
- Render configuration included in `render.yaml`

## 📁 Repository Structure

```text
agentic-ai-smart-video-editor/
│
├── app.py                         # Main Streamlit application
├── master_agent.py                # Natural-language planning agent
├── editing_agent.py               # Video editing execution engine
├── captioning_agent.py            # AI caption generation
├── summarizer_agent.py            # Video question-answering agent
├── streamlit_logger.py            # MoviePy → Streamlit progress logger
│
├── local_app.py                   # Optional local AI application
├── local_master_agent.py          # Ollama planning agent
├── local_captioning_agent.py      # Whisper transcription agent
│
├── evaluate.py                    # Evaluation utilities
├── evaluate1.py                   # Additional evaluation workflow
├── evaluation_results.csv         # Evaluation results
├── graph.ipynb                    # Evaluation / visualization notebook
│
├── assets/
│   ├── figure1_performance.png
│   ├── figure2_latency.png
│   ├── figure3_comparison.png
│   └── figure4_wer.png
│
├── docs/
│   └── IJIRT186042_PAPER.pdf      # Published paper
│
├── Dockerfile
├── render.yaml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/agentic-ai-smart-video-editor.git
cd agentic-ai-smart-video-editor
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

FFmpeg is also required for MoviePy video processing.

### 4. Configure the Gemini API key

Copy the example environment file:

```bash
cp .env.example .env
```

Then add your API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

**Never commit `.env` or your API key to GitHub.**

### 5. Run the application

```bash
streamlit run app.py
```

The Streamlit interface should open in your browser.

## 🧪 Example Prompts

Try instructions such as:

```text
Trim the video from 00:10 to 01:00.
```

```text
Create a 30-second highlight video.
```

```text
Make this video vertical in 9:16 format.
```

```text
Rotate the video by 90 degrees.
```

```text
Add English captions to the video.
```

```text
Create a short highlight and make it 9:16 for social media.
```

For video understanding:

```text
What is happening in this video?
```

```text
Describe the main actions performed in the video.
```

## 🐳 Docker

Build:

```bash
docker build -t agentic-ai-video-editor .
```

Run:

```bash
docker run -p 8501:8501 --env-file .env agentic-ai-video-editor
```

Then open:

```text
http://localhost:8501
```

## ☁️ Deployment

A Render deployment configuration is included in `render.yaml`.

Before deploying, configure the `GOOGLE_API_KEY` environment variable in the deployment platform.

> Video processing can require significant CPU/RAM and temporary storage. Production deployments should therefore use a plan appropriate for the uploaded video sizes and expected workload.

## 📊 Evaluation

The project includes evaluation scripts, results, a Jupyter notebook, and generated plots under `assets/`.

The evaluation materials are intended to support analysis of:
- system performance
- processing latency
- comparison results
- word error rate (WER)

## 📄 Publication

The research paper associated with this project is included in [`docs/IJIRT186042_PAPER.pdf`](docs/IJIRT186042_PAPER.pdf).

**Title:** *Agentic AI-Based Smart Video Editor: A Multi-Agent Framework for Automated Video Summarization and Editing*

**Publication:** International Journal of Innovative Research in Technology (IJIRT)

**Paper ID:** 186042

## 🔐 Security & Privacy

- API keys must be supplied through environment variables.
- `.env` is intentionally excluded from version control.
- Uploaded videos are processed temporarily by the application.
- Do not upload confidential or sensitive videos to third-party AI services unless you are authorized to do so.

## ⚠️ Project Status

This is an academic/research project and portfolio demonstration. Some agent operations, such as storyboard sequencing and advanced style refinement, are implemented as extensible components rather than full production-grade video-editing pipelines.

## 👤 Author

Kunj Shirish Raut

Bachelor of Engineering — Artificial Intelligence & Data Science

MCT's Rajiv Gandhi Institute of Technology, Mumbai

---

⭐ If you find the project useful, consider starring the repository.
