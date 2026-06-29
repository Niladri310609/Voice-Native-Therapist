import os
import asyncio
import json
import time
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from copilot_engine import CopilotEngine, CopilotSuggestion
from speech_processor import SpeechProcessor, SpeechSegment
from patient_data import Patient, get_patient, PATIENTS

load_dotenv()

app = FastAPI(title="Voice-Native Therapist Copilot")
app.mount("/static", StaticFiles(directory="static"), name="static")


class SessionConfig(BaseModel):
    patient_id: str
    pause_threshold_ms: int = 800


# In-memory session registry. In production, replace with Redis-backed session store.
SESSIONS: Dict[str, "CopilotSession"] = {}


class CopilotSession:
    def __init__(self, session_id: str, patient_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.engine = CopilotEngine(patient_id=patient_id)
        self.processor = SpeechProcessor(
            use_mock=os.getenv("USE_MOCK_SPEECH", "true").lower() == "true",
            on_segment=self._on_speech_segment,
        )
        self.closed = False

    def _on_speech_segment(self, segment: SpeechSegment) -> None:
        """Callback fired by the speech pipeline when a final transcript segment is available."""
        suggestions = self.engine.ingest(segment.speaker, segment.text)
        asyncio.create_task(self._emit(
            {
                "event": "transcript",
                "speaker": segment.speaker,
                "text": segment.text,
            }
        ))
        asyncio.create_task(self._emit_suggestions(suggestions))
        if self.engine.should_llm_respond(segment.speaker, segment.text):
            asyncio.create_task(self._llm_suggest(segment.speaker, segment.text))

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
                    # Demo path: typed/mock audio transcript
                    speaker = msg.get("speaker", "patient")
                    text = msg.get("text", "")
                    await self.processor.feed_text(speaker, text)

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
    return {"status": "ready", "patient_id": config.patient_id}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    initial = await ws.receive_json()
    patient_id = initial.get("patient_id", "P-001")
    session_id = initial.get("session_id", f"session-{int(time.time() * 1000)}")

    session = CopilotSession(session_id, patient_id, ws)
    SESSIONS[session_id] = session

    # Start the pause watcher concurrently.
    asyncio.create_task(session.pause_watcher())
    await session.handle_messages()

    del SESSIONS[session_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
