Video RAG Pipeline – puch_ai_hackathon
This project implements a Video Retrieval-Augmented Generation (RAG) pipeline for extracting, summarizing, and querying YouTube video content.

Overview
The pipeline:

Downloads audio from a YouTube video.

Transcribes it using Whisper.

Summarizes it using Gemini.

Stores the text chunks in Qdrant.

Lets you ask questions about the video content.

Requirements
Python 3.11

Qdrant instance running locally or remotely.

Google Gemini API key.

uv or pip for dependency installation.

Installation
Clone the repository


git clone https://github.com/yourusername/puch_ai_hackathon.git
cd puch_ai_hackathon
Create and activate a virtual environment


python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
Install dependencies


pip install -r requirements.txt
# or, if using uv:
uv pip install -r pyproject.toml
Configure environment variables
Create a .env file in the project root:


GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_if_any
Usage
Run the main pipeline:


python app/video_rag_pipeline.py
Steps:

Enter the YouTube video URL.

Wait for processing and summarization.

Ask questions about the video.

Notes
Transcription quality depends on audio clarity.

The Q&A quality depends on summarization and chunking.

Large videos may take longer to process.