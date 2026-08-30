import asyncio
import base64
import re
import os
import sys
from typing import Annotated
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from pydantic import BaseModel, Field

# Import AI functions from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.video_rag_pipeline import download_audio_from_url, transcribe_audio_local, summarize_transcript, ingest_transcript_text, query_qdrant, build_rag_prompt, call_gemini_chat

# Load env vars
load_dotenv()
TOKEN = os.getenv("AUTH_TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")
assert TOKEN, "Missing AUTH_TOKEN"
assert MY_NUMBER, "Missing MY_NUMBER"

# Auth
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        key = RSAKeyPair.generate()
        super().__init__(public_key=key.public_key)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="mcp-user", scopes=["*"])
        return None

# Tool Description
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

VIDEO_TOOL_DESC = RichToolDescription(
    description="Summarize a video and optionally answer a question about it.",
    use_when="User gives a YouTube or other video URL.",
    side_effects="Downloads video, transcribes audio, summarizes.",
)

# MCP instance
mcp = FastMCP("Puch MCP Starter", auth=SimpleBearerAuthProvider(TOKEN))

@mcp.tool(description=VIDEO_TOOL_DESC.model_dump_json())
async def video_summarizer(
    video_url: Annotated[str, Field(description="URL of video to process")],
    user_question: Annotated[str | None, Field(description="Optional question")] = None
) -> str:
    try:
        # Decode base64 video URLs if needed
        if re.match(r'^[A-Za-z0-9_-]+={0,2}$', video_url) and not video_url.startswith("http"):
            try:
                video_url = base64.urlsafe_b64decode(video_url).decode()
            except Exception:
                pass

        audio_path = download_audio_from_url(video_url)
        if not audio_path:
            return "Failed to download audio from video."

        # Transcribe in background thread
        transcript = await asyncio.to_thread(transcribe_audio_local, audio_path)
        text = transcript.get("text", "").strip()
        if not text:
            return "Transcription returned no text."

        if user_question:
            ingest_transcript_text(text, source=video_url)
            results = query_qdrant(user_question)
            if not results:
                return "No relevant context found."
            prompt = build_rag_prompt(results, user_question)
            answer = call_gemini_chat(prompt)
            return f"**Answer:** {answer}"

        summary = summarize_transcript(text)
        return f"**Summary:**\n\n{summary}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Failed: {e}"



@mcp.tool
async def validate() -> str:
    return MY_NUMBER

async def main():
    print("🚀 MCP Starter running at http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
