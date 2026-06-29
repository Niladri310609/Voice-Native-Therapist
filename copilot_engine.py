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
class SpeakerResolution:
    role: str
    confidence: float
    reason: str


@dataclass
class SessionState:
    patient_id: str
    transcript: List[Dict[str, str]] = field(default_factory=list)
    last_therapist_speech_time: float = 0.0
    pause_detected: bool = False
    risk_count: int = 0
    last_speaker: str = ""
    speaker_aliases: Dict[str, str] = field(default_factory=dict)

    def add_transcript(self, speaker: str, text: str) -> None:
        self.transcript.append({"speaker": speaker, "text": text, "ts": time.time()})
        self.last_speaker = speaker
        if speaker == "therapist":
            self.last_therapist_speech_time = time.time()
            self.pause_detected = False

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

    def resolve_speaker(
        self,
        speaker_hint: str,
        text: str,
        speaker_id: str | None = None,
        source: str = "text",
    ) -> SpeakerResolution:
        normalized_hint = (speaker_hint or "auto").strip().lower()
        if normalized_hint in {"patient", "therapist"}:
            if speaker_id:
                self.state.speaker_aliases[speaker_id] = normalized_hint
            return SpeakerResolution(
                role=normalized_hint,
                confidence=1.0,
                reason="explicit-speaker",
            )

        if speaker_id and speaker_id in self.state.speaker_aliases:
            mapped_role = self.state.speaker_aliases[speaker_id]
            return SpeakerResolution(
                role=mapped_role,
                confidence=0.95,
                reason=f"speaker-id:{speaker_id}",
            )

        lower_text = (text or "").strip().lower()
        therapist_score = 0.0
        patient_score = 0.0

        if any(trigger in lower_text for trigger in self.TRIGGER_PHRASES):
            therapist_score += 4.0

        therapist_patterns = [
            "tell me more",
            "help me understand",
            "walk me through",
            "what do we know",
            "how long",
            "when did",
            "have you",
            "can you",
            "would you",
            "let's",
        ]
        if any(pattern in lower_text for pattern in therapist_patterns):
            therapist_score += 2.0
        if "?" in text:
            therapist_score += 1.5

        patient_patterns = [
            "i feel",
            "i am",
            "i'm",
            "i've",
            "i have",
            "i can't",
            "my mood",
            "my anxiety",
            "my depression",
        ]
        if any(pattern in lower_text for pattern in patient_patterns):
            patient_score += 2.5

        patient_keywords = self._patient_context_keywords()
        if any(keyword in lower_text for keyword in patient_keywords):
            patient_score += 2.0

        if source.endswith("mic") and self.state.last_speaker:
            if self.state.last_speaker == "patient":
                therapist_score += 0.75
            else:
                patient_score += 0.75

        if therapist_score == 0.0 and patient_score == 0.0:
            fallback_role = "patient" if self.state.last_speaker != "patient" else "therapist"
            confidence = 0.45 if source.endswith("mic") else 0.35
            reason = "alternating-fallback" if self.state.last_speaker else "default-first-speaker"
            if speaker_id:
                self.state.speaker_aliases[speaker_id] = fallback_role
            return SpeakerResolution(role=fallback_role, confidence=confidence, reason=reason)

        resolved_role = "therapist" if therapist_score > patient_score else "patient"
        confidence = min(0.95, 0.55 + abs(therapist_score - patient_score) * 0.12)
        reason = f"heuristic:{source}"

        if speaker_id and abs(therapist_score - patient_score) >= 1.0:
            self.state.speaker_aliases[speaker_id] = resolved_role

        return SpeakerResolution(role=resolved_role, confidence=confidence, reason=reason)

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
        if self.state.last_speaker != "therapist":
            return []

        if self.state.last_therapist_speech_time <= 0:
            return []

        if self.state.pause_detected:
            return []

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

        self.state.pause_detected = True

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

    def _patient_context_keywords(self) -> List[str]:
        keywords = set()
        for bucket in self.RISK_KEYWORDS.values():
            keywords.update(bucket)
        keywords.update(self.DSM_HINTS.keys())

        if self.patient:
            for diagnosis in self.patient.diagnosis:
                keywords.update(re.findall(r"[a-zA-Z']{4,}", diagnosis.lower()))
            for flag in self.patient.risk_flags:
                keywords.update(re.findall(r"[a-zA-Z']{4,}", flag.lower()))
            keywords.update(trigger.lower() for trigger in self.patient.triggers)

        return sorted(keywords)
