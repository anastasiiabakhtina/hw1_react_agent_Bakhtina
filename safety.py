"""
Модуль захисних механізмів для ReAct-агента:
- обмеження максимальної кількості кроків (max_steps);
- загальний тайм-аут виконання (timeout через time.monotonic);
- детекція зациклення на однакових викликах інструментів (LoopDetector).
"""

import hashlib
import json
import time


DEFAULT_MAX_STEPS = 12
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_MAX_REPEATS = 3


class LoopDetector:
    """Детектор зациклення агента на повторюваних викликах інструментів."""

    def __init__(self, max_repeats: int = DEFAULT_MAX_REPEATS):
        if max_repeats < 2:
            raise ValueError("Параметр max_repeats має бути не меншим за 2.")
        self.max_repeats = max_repeats
        self.recent_calls: list[str] = []

    def _make_signature(self, tool_name: str, args: dict) -> str:
        """Формує детермінований MD5-хеш комбінації назви інструменту та аргументів."""
        serialized_args = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
        raw = f"{tool_name}:{serialized_args}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def check(self, tool_name: str, args: dict) -> bool:
        """Реєструє виклик та перевіряє, чи не повторюється він max_repeats разів поспіль.

        Returns:
            True, якщо виявлено зациклення; False — якщо виконання штатне.
        """
        call_hash = self._make_signature(tool_name, args)
        self.recent_calls.append(call_hash)

        if len(self.recent_calls) >= self.max_repeats:
            last_n = self.recent_calls[-self.max_repeats:]
            if len(set(last_n)) == 1:
                return True
        return False

    def reset(self) -> None:
        """Очищає історію викликів перед новим запуском."""
        self.recent_calls.clear()


class SafetyGuard:
    """Комбінований контролер безпеки ReAct-циклу (max_steps + timeout + LoopDetector)."""

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_repeats: int = DEFAULT_MAX_REPEATS,
    ):
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.loop_detector = LoopDetector(max_repeats=max_repeats)
        self.start_time = time.monotonic()

    def elapsed_seconds(self) -> float:
        """Повертає час у секундах від початку запуску агента."""
        return time.monotonic() - self.start_time

    def is_timed_out(self) -> bool:
        """Перевіряє, чи вичерпано загальний ліміт часу виконання."""
        return self.elapsed_seconds() > self.timeout_seconds

    def check_before_step(self, current_step: int) -> tuple[bool, str]:
        """Перевіряє ліміт кроків та тайм-аут перед черговою ітерацією LLM.

        Returns:
            (True, причина_зупинки) або (False, "").
        """
        if current_step >= self.max_steps:
            return (
                True,
                f"Досягнуто ліміт кроків ({self.max_steps}). Зупинка циклу для запобігання перевитраті ресурсів.",
            )

        elapsed = self.elapsed_seconds()
        if elapsed > self.timeout_seconds:
            return (
                True,
                f"Перевищено загальний тайм-аут виконання ({self.timeout_seconds:.2f} с, минуло {elapsed:.2f} с).",
            )

        return False, ""

    def check_tool_execution(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Перевіряє тайм-аут та повторюваність виклику перед запуском інструменту.

        Returns:
            (True, причина_зупинки) або (False, "").
        """
        elapsed = self.elapsed_seconds()
        if elapsed > self.timeout_seconds:
            return (
                True,
                f"Перевищено загальний тайм-аут виконання ({self.timeout_seconds:.2f} с) перед викликом '{tool_name}'.",
            )

        if self.loop_detector.check(tool_name, args):
            return (
                True,
                f"Виявлено зациклення: інструмент '{tool_name}' викликано з однаковими аргументами "
                f"{self.loop_detector.max_repeats} рази поспіль.",
            )

        return False, ""
