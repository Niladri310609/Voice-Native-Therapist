import os
import asyncio
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class SpeechSegment:
    speaker: str
    text: str
    is_final: bool = True
    source: str = "text"
    speaker_id: str | None = None


class SpeechProcessor:
    def __init__(self, use_mock: bool = True, on_segment: Optional[Callable[[SpeechSegment], None]] = None):
        self.use_mock = use_mock or os.getenv("USE_MOCK_SPEECH", "false").lower() == "true"
        self.on_segment = on_segment
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def feed_text(
        self,
        speaker: str,
        text: str,
        source: str = "text",
        speaker_id: str | None = None,
    ) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if self.on_segment:
            self.on_segment(
                SpeechSegment(
                    speaker=speaker,
                    text=cleaned,
                    is_final=True,
                    source=source,
                    speaker_id=speaker_id,
                )
            )

    async def mock_stream(self, queue: asyncio.Queue) -> None:
        pass
