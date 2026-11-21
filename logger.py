"""
Модуль структурованого JSON-логування траєкторії виконання ReAct-агента.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional


class TrajectoryLogger:
    """Фіксує кожен крок виконання вузлів LangGraph у хронологічному порядку."""

    def __init__(self, query: str = ""):
        self.query = query
        self.steps: list[dict[str, Any]] = []
        self.start_time = time.monotonic()

    @staticmethod
    def _truncate(value: Any, limit: int = 500) -> str:
        """Перетворює значення на рядок і обрізає до заданої довжини."""
        text = str(value) if value is not None else ""
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def log_step(
        self,
        step_number: int,
        node_name: str,
        input_data: Any,
        output_data: Any,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        duration_ms: int = 0,
    ) -> dict[str, Any]:
        """Додає запис про виконання одного кроку вузла до траєкторії."""
        record = {
            "step_number": step_number,
            "node_name": node_name,
            "input": self._truncate(input_data, 500),
            "output": self._truncate(output_data, 500),
            "tool_calls": tool_calls or [],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "elapsed_ms": int((time.monotonic() - self.start_time) * 1000),
        }
        self.steps.append(record)
        return record

    def to_dict(self, status: str = "completed", stop_reason: Optional[str] = None) -> dict[str, Any]:
        """Формує підсумковий словник траєкторії для серіалізації у JSON."""
        return {
            "query": self.query,
            "status": status,
            "stop_reason": stop_reason,
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
            "trajectory": self.steps,
        }

    def save(
        self,
        filepath: str = "trajectory.json",
        status: str = "completed",
        stop_reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Зберігає повну траєкторію виконання у JSON-файл."""
        payload = self.to_dict(status=status, stop_reason=stop_reason)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Траєкторію збережено у '{filepath}' (кроків: {len(self.steps)}, статус: {status})")
        return payload
