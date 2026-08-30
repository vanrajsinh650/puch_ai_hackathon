import os
import tempfile
import shutil
import uuid
import warnings
import yt_dlp
import socket
import argparse
from dotenv import load_dotenv
from pydub import AudioSegment

# Qdrant + Gemini
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

import whisper  # local open source whisper

# CONFIG 
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "video_transcripts")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
CHUNK_WORD_SIZE = int(os.getenv("CHUNK_WORD_SIZE", 160))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 30))
TOP_K = int(os.getenv("TOP_K", 4))

print("[INFO] Generative AI endpoint IP:", socket.gethostbyname("generativeai.googleapis.com"))

if not OPENROUTER_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY or GEMINI_API_KEY in .env")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

qdrant = QdrantClient(
    url=f"https://{QDRANT_HOST}",
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

genai.configure(api_key=GEMINI_API_KEY)

# Load Whisper model once (choose model size based on your hardware)
whisper_model = whisper.load_model("small")  # options: tiny, base, small, medium, large

# AUDIO UTILS 

def preprocess_audio(input_path: str, speed=1.0) -> str:
    audio = AudioSegment.from_file(input_path)
    faster_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    }).set_frame_rate(audio.frame_rate)
    trimmed = faster_audio.strip_silence(silence_thresh=-50.0, padding=100)
    processed_path = input_path.replace(".mp3", "_processed.mp3")
    trimmed.export(processed_path, format="mp3")
    return processed_path

def download_audio_from_url(video_url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio")

    def hook(d):
        if d['status'] == 'downloading':
            print(f"[DOWNLOADING] {d['_percent_str']} ETA: {d.get('eta', '?')}s")
        elif d['status'] == 'finished':
            print("[INFO] Download complete, now converting...")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "progress_hooks": [hook],
        "quiet": False,
        "noplaylist": True,
    }

    print(f"[INFO] Downloading audio from: {video_url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"[ERROR] yt_dlp failed to download: {e}")
        raise RuntimeError("Failed to download video audio.")

    final_mp3 = output_path + ".mp3"
    if not os.path.exists(final_mp3):
        raise RuntimeError(f"[ERROR] Audio file not found after download: {final_mp3}")
    print(f"[INFO] Audio file saved: {final_mp3}")

    save_dir = os.path.join(os.getcwd(), "saved_audio")
    os.makedirs(save_dir, exist_ok=True)
    saved_path = os.path.join(save_dir, f"{uuid.uuid4().hex}.mp3")
    shutil.move(final_mp3, saved_path)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return saved_path

def split_audio(file_path, chunk_length_ms=5000):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    chunk_dir = os.path.join(os.path.dirname(file_path), "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = os.path.join(chunk_dir, f"chunk_{i//chunk_length_ms}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    return chunks

def transcribe_chunk(chunk_path):
    try:
        print(f"[INFO] Transcribing chunk locally with open-source Whisper: {chunk_path}")
        result = whisper_model.transcribe(chunk_path)
        return result["text"]
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return ""

# QDRANT / GEMINI 

def init_collection():
    if QDRANT_COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )

def chunk_text(text: str, chunk_size_words=CHUNK_WORD_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    chunks, i = [], 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i + chunk_size_words]))
        i += chunk_size_words - overlap
    return chunks

def embed_texts(texts):
    embs = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [emb.astype("float32").tolist() for emb in embs]

def ingest_transcript_text(transcript: str, source: str):
    init_collection()
    chunks = chunk_text(transcript)
    embeddings = embed_texts(chunks)
    points = []
    for chunk, emb in zip(chunks, embeddings):
        points.append(qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": chunk, "source": source}
        ))
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"[INFO] Ingested {len(points)} chunks into Qdrant.")

def query_qdrant(query: str, top_k: int = TOP_K):
    vec = embed_texts([query])[0]
    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
        score_threshold=0.75
    )
    return results


def build_rag_prompt(contexts, question: str) -> str:
    context_blocks = [
        f"{i+1}. {c['payload']['text']}" for i, c in enumerate(contexts)
    ]
    return (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n" +
        "\n".join(context_blocks) +
        f"\n\nQuestion: {question}\nAnswer:"
    )

# SUMMARIZATION / QA 

def call_gemini_chat(prompt: str) -> str:
    try:
        print("\nGemini Prompt:\n" + prompt[:1500] + "\n--- END ---\n")
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini RAG QA failed: {e}")
        return "RAG-based answer failed. Please check your Gemini API key or quota."

def summarize_transcript(transcript: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        trimmed = transcript[:6000]  
        prompt = f"Summarize the following video transcript into 5 key bullet points:\n\n{trimmed}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini summarization failed: {e}")
        return "Summarization failed."

# MAIN PIPELINE 

def check_connections():
    try:
        print("[INFO] Checking Qdrant collections...")
        collections = qdrant.get_collections()
        print(f"[INFO] Qdrant connected. {len(collections.collections)} collections found.")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant: {e}")
        
def transcribe_audio_local(file_path: str) -> dict:
    try:
        processed_path = preprocess_audio(file_path)
        chunk_paths = split_audio(processed_path)

        results = []
        for i, path in enumerate(chunk_paths):
            print(f"[INFO] Transcribing chunk {i+1}/{len(chunk_paths)}...")
            results.append(transcribe_chunk(path))

        full_transcript = " ".join(results)
        return {"text": full_transcript.strip()}

    finally:
        shutil.rmtree(os.path.join(os.path.dirname(file_path), "chunks"), ignore_errors=True)


def video_to_summary(video_url: str):
    audio_path = download_audio_from_url(video_url)
    try:
        data = transcribe_audio_local(audio_path)
        transcript = data.get("text", "")
        if not transcript:
            print("[ERROR] Empty transcript")
            return

        # Summarize
        summary = summarize_transcript(transcript)
        print("\n=== SUMMARY ===\n", summary)

        # Ingest into Qdrant
        ingest_transcript_text(transcript, source=video_url)

        # RAG-based QA
        while True:
            question = input("\nAsk a question (or type 'exit'): ").strip()
            if question.lower() == "exit":
                break
            results = query_qdrant(question)
            if not results:
                print("No relevant context found.")
                continue
            prompt = build_rag_prompt(results, question)
            answer = call_gemini_chat(prompt)
            print("Gemini Answer:", answer)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Video URL to transcript summary with RAG QA")
    parser.add_argument("video_url", help="YouTube or video URL to process")
    args = parser.parse_args()

    check_connections()
    video_to_summary(args.video_url)

if __name__ == "__main__":
    main()
