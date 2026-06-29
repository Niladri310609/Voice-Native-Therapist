import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from openai import AsyncAzureOpenAI

from patient_data import Patient, get_patient


@dataclass
class CopilotSuggestion:
    type: str  # "suggestion", "history", "dsm", "risk", "summary"
    title: str
    body: str
    urgency: str = "low"  # low | medium | high | critical


@dataclass
class SessionState:
    patient_id: str
    transcript: List[Dict[str, str]] = field(default_factory=list)
    last_therapist_speech_time: float = 0.0
    pause_detected: bool = False
    risk_count: int = 0

    def add_transcript(self, speaker: str, text: str) -> None:
        self.transcript.append({"speaker": speaker, "text": text, "ts": time.time()})
        if speaker == "therapist":
            self.last_therapist_speech_time = time.time()

    def get_full_context(self, max_turns: int = 10) -> str:
        recent = self.transcript[-max_turns:]
        return "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in recent)

    def get_last_patient_utterance(self) -> str:
        for t in reversed(self.transcript):
            if t["speaker"] == "patient":
                return t["text"]
        return ""


class CopilotEngine:
    # Phrases that cause the copilot to surface a response.
    TRIGGER_PHRASES = [
        "help me think",
        "what do we know",
        "what's the risk",
        "any red flags",
        "suggest",
        "recommend",
        "history",
    ]

    # Risk keywords mapped to escalation level.
    RISK_KEYWORDS = {
        "critical": ["suicide", "kill myself", "end my life", "want to die", "hurt myself"],
        "high": ["hopeless", "worthless", "burden", "can't go on", "no point", "overdose"],
        "medium": ["panic attack", "can't breathe", "drink", "alcohol", "missed dose"],
    }

    DSM_HINTS = {
        "depressed": "Major Depressive Disorder (F32.9): depressed mood, anhedonia, fatigue.",
        "anxious": "Generalized Anxiety Disorder (F41.1): excessive worry, restlessness, difficulty concentrating.",
        "can't focus": "ADHD (F90.2): inattention, disorganization, impulsivity.",
        "can't sleep": "Insomnia Disorder (G47.0): difficulty initiating/maintaining sleep.",
        "flashback": "PTSD (F43.10): intrusive memories, re-experiencing trauma.",
        "hearing voices": "Schizophrenia Spectrum (F20.9): hallucinations require screening.",
    }

    def __init__(self, patient_id: str, pause_threshold_ms: int = 800):
        self.patient: Optional[Patient] = get_patient(patient_id)
        self.state = SessionState(patient_id=patient_id)
        self.pause_threshold_ms = pause_threshold_ms
        self.last_emit_time = 0.0

        self.llm_enabled = False
        self.client: Optional[AsyncAzureOpenAI] = None
        self.model = ""

        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if api_key and endpoint:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=endpoint,
            )
            self.model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
            self.llm_enabled = True

    def ingest(self, speaker: str, text: str) -> List[CopilotSuggestion]:
        """Process a new transcript utterance and return suggestions."""
        self.state.add_transcript(speaker, text)
        suggestions: List[CopilotSuggestion] = []

        lower_text = text.lower()

        # 1. Risk detection (always check for patient utterances)
        if speaker == "patient":
            risk_suggestions = self._detect_risk(lower_text)
            suggestions.extend(risk_suggestions)

        # 2. Trigger phrase detection (therapist asked for help)
        if speaker == "therapist" and any(trigger in lower_text for trigger in self.TRIGGER_PHRASES):
            if not self.llm_enabled:
                suggestions.extend(self._build_contextual_response())

        # 3. DSM-aligned observation for patient utterances
        if speaker == "patient":
            dsm_suggestions = self._detect_dsm_observations(lower_text)
            suggestions.extend(dsm_suggestions)

        # 4. Silence/pause detection handled separately by heartbeat.
        return suggestions

    def should_llm_respond(self, speaker: str, text: str) -> bool:
        return self.llm_enabled and speaker == "therapist" and any(trigger in text.lower() for trigger in self.TRIGGER_PHRASES)

    async def llm_suggest(self, speaker: str, text: str) -> List[CopilotSuggestion]:
        if not self.client or not self.model:
            return []

        patient = self.patient
        context = self.state.get_full_context(max_turns=8)

        system_message = (
            "You are a clinical copilot assisting a therapist during a live session. "
            "Return ONLY a JSON object with keys: title, body, urgency (low/medium/high/critical), type (suggestion/history/dsm/risk/summary). "
            "Do not include markdown, code fences, or any extra text."
        )

        user_parts = []
        if patient:
            user_parts.append(
                f"Patient: {patient.name}, age {patient.age}. "
                f"Diagnoses: {', '.join(patient.diagnosis)}. "
                f"Medications: {', '.join(patient.medications)}. "
                f"Risk flags: {', '.join(patient.risk_flags) or 'none'}. "
                f"History: {patient.history_summary}"
            )
            if patient.recent_notes:
                user_parts.append("Recent notes:\n" + "\n".join(patient.recent_notes[-2:]))

        user_parts.append(f"Recent transcript:\n{context}")
        user_parts.append(f"Latest {speaker} utterance: {text}")

        user_message = (
            "Based on the patient record and the recent conversation, provide one concise, actionable suggestion for the therapist. "
            + "\n\n".join(user_parts)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.strip("`").strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()

            data = json.loads(content)
            return [CopilotSuggestion(
                type=data.get("type", "suggestion"),
                title=data.get("title", "LLM suggestion"),
                body=data.get("body", content),
                urgency=data.get("urgency", "low"),
            )]
        except Exception:
            return []

    def on_pause(self) -> List[CopilotSuggestion]:
        """Called when a pause is detected; return a quick summary or next-step cue."""
        elapsed = (time.time() - self.state.last_therapist_speech_time) * 1000
        if elapsed < self.pause_threshold_ms:
            return []

        # Avoid spamming: only emit once per pause window.
        if time.time() - self.last_emit_time < 2.0:
            return []
        self.last_emit_time = time.time()

        patient = self.patient
        last_patient = self.state.get_last_patient_utterance()
        if not patient or not last_patient:
            return []

        return [CopilotSuggestion(
            type="summary",
            title="Pause detected — suggested next move",
            body=(f"Patient mentioned: '{last_patient}'. "
                  f"Consider validating emotion and asking an open question. "
                  f"Risk flags: {', '.join(patient.risk_flags) or 'none'}."),
            urgency="low",
        )]

    def _detect_risk(self, text: str) -> List[CopilotSuggestion]:
        suggestions: List[CopilotSuggestion] = []
        for level, keywords in self.RISK_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    self.state.risk_count += 1
                    if level == "critical":
                        suggestions.append(CopilotSuggestion(
                            type="risk",
                            title="CRITICAL RISK FLAG",
                            body=(f"Keyword '{kw}' detected. Perform immediate safety assessment "
                                  "(Columbia Scale / lethal means). Escalate per protocol."),
                            urgency="critical",
                        ))
                    elif level == "high":
                        suggestions.append(CopilotSuggestion(
                            type="risk",
                            title="High-risk language",
                            body=(f"Keyword '{kw}' detected. Explore intent, plan, means, and "
                                  "protective factors. Document in EHR."),
                            urgency="high",
                        ))
                    else:
                        suggestions.append(CopilotSuggestion(
                            type="risk",
                            title="Risk signal",
                            body=f"Keyword '{kw}' may indicate decompensation or relapse.",
                            urgency="medium",
                        ))
        return suggestions

    def _detect_dsm_observations(self, text: str) -> List[CopilotSuggestion]:
        suggestions: List[CopilotSuggestion] = []
        for symptom, dsm_note in self.DSM_HINTS.items():
            if symptom in text:
                suggestions.append(CopilotSuggestion(
                    type="dsm",
                    title="DSM-aligned observation",
                    body=dsm_note,
                    urgency="low",
                ))
        return suggestions

    def _build_contextual_response(self) -> List[CopilotSuggestion]:
        if not self.patient:
            return [CopilotSuggestion(
                type="history",
                title="Patient unknown",
                body="No patient record loaded for this session.",
                urgency="low",
            )]

        patient = self.patient
        context = self.state.get_full_context(max_turns=6)

        suggestions = [
            CopilotSuggestion(
                type="history",
                title="Relevant patient history",
                body=(f"{patient.name}, {patient.age} — {'; '.join(patient.diagnosis)}. "
                      f"Meds: {'; '.join(patient.medications)}. "
                      f"Summary: {patient.history_summary}"),
                urgency="low",
            ),
            CopilotSuggestion(
                type="suggestion",
                title="Contextual suggestion",
                body=(f"Based on recent context:\n{context}\n\n"
                      f"Consider asking: 'How has your mood been since starting {patient.medications[0]}?'"),
                urgency="low",
            ),
        ]

        if patient.recent_notes:
            suggestions.append(CopilotSuggestion(
                type="history",
                title="Recent notes",
                body="\n".join(patient.recent_notes[-2:]),
                urgency="low",
            ))

        return suggestions
