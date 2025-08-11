import os
import tempfile
import shutil
import uuid
import warnings
import yt_dlp
import torch
import socket
from dotenv import load_dotenv
from transformers import pipeline
from pydub import AudioSegment

# Qdrant + Gemini
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(socket.gethostbyname("generativeai.googleapis.com"))



if not OPENROUTER_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY or GEMINI_API_KEY in .env")

# Models 
asr_pipeline = pipeline("automatic-speech-recognition",
                        model="Gwenn-LR/wisper-small-dv",
                        device=0,
                        torch_dtype=torch.float16,
                        generate_kwargs={"language": "en"})

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

qdrant = QdrantClient(
    url=f"https://{QDRANT_HOST}",
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)


genai.configure(api_key=GEMINI_API_KEY)

# AUDIO UTILS 

def preprocess_audio(input_path: str, speed=2.0) -> str:
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
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": False,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    final_mp3 = output_path + ".mp3"
    if not os.path.exists(final_mp3):
        raise RuntimeError(f"Audio file not found: {final_mp3}")
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

# TRANSCRIPTION 

def transcribe_chunk(chunk_path):
    try:
        result = asr_pipeline(chunk_path)
        return result.get("text", "")
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return ""


def transcribe_audio_local(file_path: str) -> dict:
    try:
        processed_path = preprocess_audio(file_path)
        chunk_paths = split_audio(processed_path)

        results = [transcribe_chunk(path) for path in chunk_paths]

        full_transcript = " ".join(results)
        return {"text": full_transcript.strip()}

    finally:
        shutil.rmtree(os.path.join(os.path.dirname(file_path), "chunks"), ignore_errors=True)


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
    results = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=top_k)
    return [{"id": r.id, "payload": r.payload} for r in results]

def build_rag_prompt(contexts, question: str) -> str:
    context_blocks = [
        f"Context {i+1}:\n{c['payload']['text']}"
        for i, c in enumerate(contexts)
    ]
    return (
        "You are a professional, respectful, and precise assistant. "
        "Carefully read the provided context snippets and answer the user's question clearly, "
        "politely, and directly. Use perfect grammar and a courteous tone. "
        "If the context provides only partial clues, use reasoning and inference to give the most relevant possible answer. "
        "Only if there is truly no relevant information at all should you respond exactly with: "
        "'I'm sorry, but the provided information does not contain a clear answer to your question.'\n\n"
        + "\n\n---\n\n".join(context_blocks) +
        f"\n\nQUESTION: {question}\n"
    )


# SUMMARIZATION / QA 

def summarize_transcript(transcript: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Summarize the following text into 5 concise bullet points:\n\n{transcript}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini summarization failed: {e}")
        return "Summarization failed. Please check your Gemini API key or quota."
    
def call_gemini_chat(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini RAG QA failed: {e}")
        return "RAG-based answer failed. Please check your Gemini API key or quota."

# MAIN PIPELINE 

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

def check_connections():
    try:
        print("[INFO] Checking Qdrant collections...")
        collections = qdrant.get_collections()
        print(f"[INFO] Qdrant connected. {len(collections.collections)} collections found.")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant: {e}")

    try:
        print("[INFO] Checking Gemini summarization...")
        test = summarize_transcript("This is a test sentence for summarization.")
        print("[INFO] Gemini response:", test)
    except Exception as e:
        print(f"[ERROR] Gemini test failed: {e}")

# Call this at the top of __main__
if __name__ == "__main__":
    check_connections()
    url = input("Enter video URL: ").strip()
    video_to_summary(url)
