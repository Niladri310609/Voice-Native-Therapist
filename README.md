# Voice-Native Therapist Copilot

A passive, ambient AI assistant for therapy sessions. It listens in the background, transcribes speech via **Azure Speech Service**, detects therapist pauses or trigger phrases, and surfaces real-time suggestions in a chatbox:

- Relevant patient history
- DSM-aligned clinical observations
- Risk flags (suicidal / self-harm / relapse language)
- Contextual next-step recommendations

Designed for **sub-500 ms response latency** with full context awareness.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Therapy Room                            │
│  ┌──────────────┐        ┌──────────────┐                       │
│  │  Microphone  │        │   Browser    │  (WebSocket)          │
│  │  (therapist) │        │  copilot UI  │                       │
│  └──────┬───────┘        └──────┬───────┘                       │
└─────────┼───────────────────────┼───────────────────────────────┘
          │                       │
          │    PCM audio stream   │
          │  ┌──────────────────┐ │
          └─►│  Azure Speech    │─┘
             │  Service (STT)   │
             │  + diarization   │
             └────────┬─────────┘
                      │ final transcript
                      ▼
             ┌──────────────────┐
             │  FastAPI server  │  WebSocket session
             │  CopilotEngine   │  · context window
             │  · risk detection│  · pause watcher
             │  · DSM hints     │  · patient lookup
             │  · suggestions   │
             └────────┬─────────┘
                      │ JSON suggestion
                      ▼
             ┌──────────────────┐
             │   Browser UI     │  chatbox (history, DSM, risk)
             └──────────────────┘
```

---

## 2. Core Components

| File | Role |
|------|------|
| `app.py` | FastAPI backend: WebSocket sessions, REST endpoints, static file serving |
| `copilot_engine.py` | Business logic: transcript buffer, risk detection, DSM observations, suggestions |
| `speech_processor.py` | Azure Speech SDK wrapper + mock fallback for offline demos |
| `patient_data.py` | Mock patient records (diagnoses, meds, risk flags, history) |
| `static/index.html` | Single-page demo UI with WebSocket client |

---

## 3. How It Works

### 3.1 Trigger Detection
- **Pause watcher**: background loop detects therapist silence (> 800 ms) and nudges the clinician with a next-step suggestion.
- **Trigger phrases**: therapist can say "help me think", "what's the risk", "suggest", etc. The copilot immediately pulls patient history and a contextual recommendation.

### 3.2 Risk Detection
Keyword-based triage runs on every patient utterance:

| Level | Example keywords | Action |
|-------|------------------|--------|
| `critical` | suicide, kill myself, end my life | Immediate safety assessment |
| `high` | hopeless, worthless, burden, no point | Explore intent, plan, means |
| `medium` | panic, alcohol, missed dose | Decompensation / relapse signal |

### 3.3 Context Awareness
- A sliding transcript window keeps the last N turns in memory.
- Patient history (diagnoses, medications, recent notes) is loaded at session start.
- Suggestions combine both live transcript + EHR-like data.

### 3.4 Latency Design
- WebSocket is a persistent full-duplex connection — no HTTP request/response overhead.
- All processing happens in-memory in the same event loop.
- Suggestions are emitted immediately after a final transcript segment arrives.

---

## 4. Scalability Design

This repo is a **single-node demo**. The production-ready architecture would scale as follows:

### 4.1 Session State
- **Current**: in-memory `SESSIONS` dict.
- **Production**: Redis-backed session store with TTL per session.

### 4.2 Speech Layer
- **Current**: one recognizer per session via Azure Speech SDK.
- **Production**: dedicated Speech-to-Text microservice pool, or Azure Speech Containers deployed in the same region for lower latency.

### 4.3 Suggestion Engine
- **Current**: rule-based keyword + mock LLM.
- **Production**: cacheable patient embeddings + a lightweight LLM (Azure OpenAI) with streaming, batched inference, and prompt templates.

### 4.4 Deployment
- Containerized with Docker.
- Kubernetes with horizontal pod autoscaling based on active WebSocket connections.
- Regional deployment close to Azure Speech endpoints.

### 4.5 Security & Compliance
- End-to-end encryption (TLS/WSS).
- Audio/transcripts encrypted at rest.
- Role-based access, audit logs.
- HIPAA/BAA considerations with Azure.

---

## 5. Run the Demo

### 5.1 Install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Configure (optional for live Azure Speech)

```bash
cp .env.example .env
# Edit .env with your Azure Speech key and region
# Set USE_MOCK_SPEECH=true to run without Azure keys
```

### 5.3 Start the server

```bash
python app.py
```

Open `http://localhost:8000` in your browser.

### 5.4 Demo flow

1. Click **Start Session**.
2. Choose a patient (Sarah Thompson or James Chen).
3. In the transcript box, type as the **patient**:
   - `I feel worthless and can't go on`
   - → Copilot will show a **critical/high risk** flag and a DSM observation.
4. Type as the **therapist**:
   - `help me think about the risk`
   - → Copilot will surface patient history, recent notes, and a contextual suggestion.
5. Pause for ~1 second after the therapist speaks → the copilot will emit a pause-based nudge.

---

## 6. Interview Walkthrough Script

Use this exact narrative to explain the project in an interview.

> **"I built a Voice-Native Therapist Copilot. The core idea is that the therapist doesn't type or click during a session — the copilot listens passively, understands context, and surfaces only the most useful information in real time."**

### 6.1 High-level pitch (30 seconds)

- **Input**: live audio from the therapy room.
- **Processing**: Azure Speech Service converts audio to text; a FastAPI backend maintains session context.
- **Output**: real-time suggestions in a chatbox — history, DSM observations, risk flags.
- **Goal**: sub-500 ms latency, zero disruption to the session.

### 6.2 Technical deep dive (2 minutes)

**Latency**
> "I used a persistent WebSocket between the browser and the FastAPI server. The moment Azure Speech returns a final transcript, the copilot engine runs in-memory and pushes a suggestion back. There's no HTTP round-trip overhead."

**Context Awareness**
> "The backend keeps a sliding transcript window and a loaded patient record. When the therapist pauses or asks for help, the engine combines live conversation + EHR data to generate a relevant response, rather than a generic answer."

**Scalability**
> "The demo uses in-memory sessions, but the architecture is designed to scale horizontally. In production, session state moves to Redis, speech processing becomes a pooled service, and the suggestion engine is a separate inferencing layer that can be batched and cached."

### 6.3 Safety & Clinical Value

> "Risk language is detected immediately and escalated by severity. The system never replaces the clinician — it augments them. For example, if a patient says 'I feel worthless,' the copilot flags it as high-risk and suggests exploring intent, plan, and means, while also showing the patient's prior suicide attempt history."

### 6.4 Trade-offs

> "For the demo, I used a rule-based suggestion engine because it's fast and deterministic. In production, I would layer an LLM on top for more nuanced phrasing, but keep the rule-based risk detection as a hard safety guardrail."

---

## 7. File Map

```
Voice-Native-Therapist/
├── app.py                 # FastAPI server + WebSocket sessions
├── copilot_engine.py      # Context, risk, DSM, suggestions
├── speech_processor.py    # Azure Speech SDK + mock fallback
├── patient_data.py        # Mock EHR records
├── static/index.html      # Demo UI
├── requirements.txt       # Python dependencies
├── .env.example           # Azure key template
└── README.md              # This file
```

---

## 8. Production Next Steps

- Replace mock speech path with real-time Azure Speech push-stream audio.
- Add speaker diarization to distinguish therapist vs patient automatically.
- Integrate with an EHR API for live patient records.
- Add LLM-based suggestion generation with streaming responses.
- Implement Redis session persistence and horizontal scaling.
- Add audit logging and consent management.
