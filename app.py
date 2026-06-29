import os
import asyncio
import time
from typing import Dict
from urllib import request, error

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from copilot_engine import CopilotEngine
from speech_processor import SpeechProcessor, SpeechSegment
from patient_data import get_patient, PATIENTS

load_dotenv()

def _cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,https://theris.netlify.app",
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


app = FastAPI(title="Voice-Native Therapist Copilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


class SessionConfig(BaseModel):
    patient_id: str
    pause_threshold_ms: int = 800
    language: str = "en-US"


# In-memory session registry. In production, replace with Redis-backed session store.
SESSIONS: Dict[str, "CopilotSession"] = {}


class CopilotSession:
    def __init__(self, session_id: str, patient_id: str, websocket: WebSocket, pause_threshold_ms: int = 800):
        self.session_id = session_id
        self.websocket = websocket
        self.engine = CopilotEngine(patient_id=patient_id, pause_threshold_ms=pause_threshold_ms)
        self.processor = SpeechProcessor(
            use_mock=os.getenv("USE_MOCK_SPEECH", "true").lower() == "true",
            on_segment=self._on_speech_segment,
        )
        self.closed = False

    def _on_speech_segment(self, segment: SpeechSegment) -> None:
        """Callback fired by the speech pipeline when a final transcript segment is available."""
        speaker = self.engine.resolve_speaker(
            segment.speaker,
            segment.text,
            speaker_id=segment.speaker_id,
            source=segment.source,
        )
        suggestions = self.engine.ingest(speaker.role, segment.text)
        asyncio.create_task(self._emit(
            {
                "event": "transcript",
                "speaker": speaker.role,
                "text": segment.text,
                "source": segment.source,
                "speaker_confidence": round(speaker.confidence, 2),
                "speaker_reason": speaker.reason,
            }
        ))
        asyncio.create_task(self._emit_suggestions(suggestions))
        if self.engine.should_llm_respond(speaker.role, segment.text):
            asyncio.create_task(self._llm_suggest(speaker.role, segment.text))

    async def _emit(self, payload: dict) -> None:
        if not self.closed:
            await self.websocket.send_json(payload)

    async def _emit_suggestions(self, suggestions: list) -> None:
        for s in suggestions:
            await self._emit(
                {
                    "event": "suggestion",
                    "type": s.type,
                    "title": s.title,
                    "body": s.body,
                    "urgency": s.urgency,
                }
            )

    async def _llm_suggest(self, speaker: str, text: str) -> None:
        suggestions = await self.engine.llm_suggest(speaker, text)
        if suggestions:
            await self._emit_suggestions(suggestions)

    async def handle_messages(self) -> None:
        await self.processor.start()
        try:
            while True:
                msg = await self.websocket.receive_json()
                event = msg.get("event")

                if event == "text":
                    # Automated path: the frontend can stream mic transcripts without manual role labels.
                    speaker = msg.get("speaker", "auto")
                    text = msg.get("text", "")
                    source = msg.get("source", "text")
                    speaker_id = msg.get("speaker_id")
                    await self.processor.feed_text(
                        speaker,
                        text,
                        source=source,
                        speaker_id=speaker_id,
                    )

                elif event == "audio_chunk":
                    # Production path: binary PCM audio from client microphone
                    # In a real deployment, forward to Azure Speech SDK push stream.
                    pass

                elif event == "ping":
                    await self._emit({"event": "pong", "ts": time.time()})

        except WebSocketDisconnect:
            self.closed = True
            await self.processor.stop()

    async def pause_watcher(self) -> None:
        """Background loop that checks for therapist silence and emits helpful nudges."""
        while not self.closed:
            await asyncio.sleep(0.5)
            suggestions = self.engine.on_pause()
            if suggestions:
                await self._emit_suggestions(suggestions)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/patients")
async def list_patients() -> list:
    return [
        {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "diagnosis": p.diagnosis,
            "risk_flags": p.risk_flags,
        }
        for p in PATIENTS.values()
    ]


@app.get("/api/patients/{patient_id}")
async def get_patient_api(patient_id: str) -> dict:
    patient = get_patient(patient_id)
    if not patient:
        return {"error": "Patient not found"}
    return {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "diagnosis": patient.diagnosis,
        "medications": patient.medications,
        "risk_flags": patient.risk_flags,
        "history_summary": patient.history_summary,
        "recent_notes": patient.recent_notes,
    }


@app.post("/api/session")
async def create_session(config: SessionConfig) -> dict:
    # Session creation is a REST call; the actual WebSocket is upgraded from /ws.
    return {
        "status": "ready",
        "patient_id": config.patient_id,
        "pause_threshold_ms": config.pause_threshold_ms,
        "speech": speech_runtime_config(),
    }


@app.get("/api/speech/runtime")
async def get_speech_runtime() -> dict:
    return speech_runtime_config()


@app.get("/api/speech/token")
async def get_speech_token() -> dict:
    if not speech_runtime_config()["enabled"]:
        return {
            "enabled": False,
            "provider": "browser",
            "message": "Azure Speech credentials are not configured on the server.",
        }

    try:
        token = await asyncio.to_thread(fetch_speech_token)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "enabled": True,
        "provider": "azure",
        "token": token,
        "region": os.getenv("AZURE_SPEECH_REGION", "").strip(),
        "language": os.getenv("AZURE_SPEECH_LANGUAGE", "en-US").strip() or "en-US",
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    initial = await ws.receive_json()
    patient_id = initial.get("patient_id", "P-001")
    session_id = initial.get("session_id", f"session-{int(time.time() * 1000)}")
    pause_threshold_ms = int(initial.get("pause_threshold_ms", 800))

    session = CopilotSession(session_id, patient_id, ws, pause_threshold_ms=pause_threshold_ms)
    SESSIONS[session_id] = session

    # Start the pause watcher concurrently.
    asyncio.create_task(session.pause_watcher())
    await session.handle_messages()

    del SESSIONS[session_id]


def speech_runtime_config() -> dict:
    region = os.getenv("AZURE_SPEECH_REGION", "").strip()
    key = os.getenv("AZURE_SPEECH_KEY", "").strip()
    return {
        "enabled": bool(region and key),
        "provider": "azure" if region and key else "browser",
        "region": region,
        "language": os.getenv("AZURE_SPEECH_LANGUAGE", "en-US").strip() or "en-US",
    }


def fetch_speech_token() -> str:
    key = os.getenv("AZURE_SPEECH_KEY", "").strip()
    region = os.getenv("AZURE_SPEECH_REGION", "").strip()
    if not key or not region:
        raise RuntimeError("Azure Speech credentials are not configured.")

    token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    req = request.Request(
        token_url,
        method="POST",
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": "0",
        },
        data=b"",
    )
    try:
        with request.urlopen(req, timeout=10) as response:
            return response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Azure Speech token request failed: {exc.code} {detail}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
