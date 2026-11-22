"""
Модуль запуску набору тест-кейсів різної складності для ReAct-агента
та збереження структурованих результатів у test_results.json.
"""

import json
import time
from agent import run_agent


# Набір тест-кейсів для доменної задачі фінансового контролю
TEST_CASES = [
    {
        "id": "TC-001",
        "complexity": "simple",
        "query": "Які витрати були зафіксовані у вересні 2026 року (2026-09)?",
        "expected": (
            "Виклик get_financial_records(period='2026-09', record_type='expense'); "
            "повернення витрат: salaries (110 000 UAH) та software (18 500 UAH), разом 128 500 UAH."
        ),
        "max_steps": 10,
        "timeout_seconds": 60.0,
    },
    {
        "id": "TC-002",
        "complexity": "medium",
        "query": "Порівняй чистий прибуток і маржинальність за 2026-09 та 2026-10 при ставці податку 5%.",
        "expected": (
            "Отримання записів за 2026-09 і 2026-10 через get_financial_records та розрахунок "
            "прибутку через calculate_profit_metrics для обох місяців; порівняння чистого прибутку "
            "(105 925.00 UAH у вересні проти 147 250.00 UAH у жовтні) та маржі (44.14% проти 41.48%)."
        ),
        "max_steps": 10,
        "timeout_seconds": 60.0,
    },
    {
        "id": "TC-003",
        "complexity": "complex",
        "query": (
            "Зафіксуй новий дохід за 2026-11 на суму 65000 грн у категорію consulting з описом "
            "'Впровадження AI-агента', а потім перевір виконання бюджету витрат за 2026-11 при ліміті "
            "150000 грн та порахуй оновлений чистий прибуток за листопад при податку 5%."
        ),
        "expected": (
            "Послідовний виклик record_transaction, analyze_expense_budget, get_financial_records "
            "та calculate_profit_metrics; підтвердження запису транзакції, статусу бюджету (140 000 / 150 000 UAH — "
            "у межах ліміту) та розрахунок чистого прибутку за 2026-11 (194 750.00 UAH при доході 345 000 UAH)."
        ),
        "max_steps": 12,
        "timeout_seconds": 90.0,
    },
    {
        "id": "TC-004",
        "complexity": "complex_validation",
        "query": (
            "Перевір витрати за 2026-10 у категорії 'crypto', а також перевір загальний бюджет "
            "витрат за 2026-10 при встановленому ліміті 180000 грн."
        ),
        "expected": (
            "Обробка помилки валідації Pydantic для неіснуючої категорії 'crypto' (без падіння графа), "
            "виклик analyze_expense_budget(period='2026-10', budget_limit_uah=180000) та виявлення "
            "перевитрати бюджету на 20 000.00 UAH."
        ),
        "max_steps": 10,
        "timeout_seconds": 60.0,
    },
    {
        "id": "TC-005",
        "complexity": "guardrail_max_steps",
        "query": (
            "Проведи повний квартальний аудит за 2026-09, 2026-10 та 2026-11: отримай усі транзакції, "
            "порахуй чистий прибуток для кожного місяця окремо та перевір бюджет 150000 грн для кожного місяця."
        ),
        "expected": (
            "Спрацювання захисного обмеження max_steps=1 після першої ітерації циклу; "
            "коректна зупинка зі статусом stopped_by_guard та повернення часткової відповіді."
        ),
        "max_steps": 1,
        "timeout_seconds": 60.0,
    },
    {
        "id": "TC-006",
        "complexity": "guardrail_timeout",
        "query": "Перевір бюджет витрат за 2026-10 при ліміті 180000 грн та порахуй чистий прибуток.",
        "expected": (
            "Спрацювання захисного обмеження timeout_seconds=0.001 ще до першого виклику LLM; "
            "коректна зупинка зі статусом stopped_by_guard (timeout_reached=True) та повернення часткової відповіді."
        ),
        "max_steps": 10,
        "timeout_seconds": 0.001,
    },
]


def truncate_text(text: str, limit: int = 500) -> str:
    """Обмежує довжину текстового поля для компактного збереження у JSON."""
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def run_test_suite(output_filepath: str = "test_results.json") -> list[dict]:
    """Запускає всі тест-кейси та зберігає результати у JSON-файл."""
    results: list[dict] = []

    for tc in TEST_CASES:
        print("\n" + "=" * 70)
        print(f"Запуск {tc['id']} [{tc['complexity']}]: {tc['query'][:75]}...")
        print("=" * 70)

        start_mono = time.monotonic()
        try:
            agent_output = run_agent(
                query=tc["query"],
                max_steps=tc["max_steps"],
                timeout_seconds=tc["timeout_seconds"],
                verbose=True,
                save_trajectory_path=None,
            )

            structured = agent_output.get("structured_response") or {}
            actual_answer = structured.get("answer", "")
            stop_reason = agent_output.get("stop_reason") or ""

            max_steps_reached = "ліміт кроків" in stop_reason.lower()
            timeout_reached = "тайм-аут" in stop_reason.lower()
            status = agent_output.get("status", "completed")
            steps = agent_output.get("step_count", 0)
            tool_calls = agent_output.get("tool_calls", [])
            confidence = structured.get("confidence")
            sources = structured.get("sources", [])

        except Exception as exc:
            actual_answer = f"Error: {type(exc).__name__}: {exc}"
            stop_reason = str(exc)
            max_steps_reached = False
            timeout_reached = False
            status = "error"
            steps = -1
            tool_calls = []
            confidence = 0.0
            sources = []

        elapsed_ms = int((time.monotonic() - start_mono) * 1000)

        results.append(
            {
                "test_id": tc["id"],
                "complexity": tc["complexity"],
                "query": tc["query"],
                "expected": tc["expected"],
                "actual": truncate_text(actual_answer, 500),
                "status": status,
                "steps": steps,
                "tool_calls": tool_calls,
                "sources": sources,
                "confidence": confidence,
                "elapsed_ms": elapsed_ms,
                "max_steps_reached": max_steps_reached,
                "timeout_reached": timeout_reached,
                "stop_reason": stop_reason if stop_reason else None,
            }
        )

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Тестування завершено. Збережено {len(results)} кейсів у '{output_filepath}':")
    for r in results:
        print(
            f"  - {r['test_id']} ({r['complexity']}): status={r['status']}, "
            f"steps={r['steps']}, tools={r['tool_calls']}, time={r['elapsed_ms']}ms, "
            f"max_steps_reached={r['max_steps_reached']}"
        )

    return results


if __name__ == "__main__":
    run_test_suite()
