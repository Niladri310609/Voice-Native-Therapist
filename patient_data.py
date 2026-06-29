from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Patient:
    id: str
    name: str
    age: int
    diagnosis: List[str]
    medications: List[str]
    risk_flags: List[str]
    history_summary: str
    recent_notes: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)


PATIENTS: Dict[str, Patient] = {
    "P-001": Patient(
        id="P-001",
        name="Sarah Thompson",
        age=34,
        diagnosis=["Major Depressive Disorder (F32.9)", "Generalized Anxiety Disorder (F41.1)"],
        medications=["Sertraline 50mg daily", "Clonazepam 0.5mg PRN"],
        risk_flags=["previous suicide attempt (2023)", "recent job loss"],
        history_summary="Patient has recurrent depressive episodes, anxiety since childhood, and a history of self-harm. Recently laid off and reporting increased hopelessness.",
        recent_notes=[
            "2026-05-10: Reports insomnia, weight loss, and difficulty concentrating.",
            "2026-06-15: Mentioned feeling like a burden to family. No active plan disclosed.",
        ],
        triggers=["suicide", "kill myself", "hopeless", "worthless", "burden", "can't go on"],
    ),
    "P-002": Patient(
        id="P-002",
        name="James Chen",
        age=29,
        diagnosis=["ADHD (F90.2)", "Mild Alcohol Use Disorder (F10.10)"],
        medications=["Methylphenidate 20mg daily"],
        risk_flags=["substance use", "medication non-adherence"],
        history_summary="Adult ADHD, struggles with impulse control and executive function. Has history of binge drinking on weekends, missed 2 doses last month.",
        recent_notes=[
            "2026-05-20: Discussed difficulty meeting deadlines at work.",
            "2026-06-22: Reported increased alcohol use after argument with partner.",
        ],
        triggers=["drink", "alcohol", "overdose", "missed dose", "can't focus"],
    ),
}


def get_patient(patient_id: str) -> Patient | None:
    return PATIENTS.get(patient_id)
