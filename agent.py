"""
Модуль ReAct-агента на базі LangGraph (StateGraph) з циклом LLM–tools–LLM,
захисними механізмами (max_steps, timeout, LoopDetector), логуванням траєкторії
та структурованим виводом (Structured Output).
"""

import os
import time
from typing import Annotated, Literal, Optional, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from logger import TrajectoryLogger
from safety import DEFAULT_MAX_REPEATS, DEFAULT_MAX_STEPS, DEFAULT_TIMEOUT_SECONDS, SafetyGuard
from tools import FINANCIAL_TOOLS

load_dotenv()

DEFAULT_MODEL = "google/gemini-3.8-flash"

SYSTEM_PROMPT = """Ти — AI-асистент з фінансового контролю та управлінського обліку.

МЕТА:
Допомагати користувачу контролювати доходи, витрати, чистий прибуток та дотримання бюджету.

ПОЛІТИКИ ВИКОРИСТАННЯ ІНСТРУМЕНТІВ:
1. Щоб дізнатися доходи, витрати чи транзакції за місяць (формат YYYY-MM) — викликай `get_financial_records`.
2. Щоб порахувати операційний/чистий прибуток, податок і рентабельність (маржу) — викликай `calculate_profit_metrics`. Якщо суми ще невідомі, спочатку отримай їх через `get_financial_records`.
3. Щоб додати нову операцію до реєстру — викликай `record_transaction`.
4. Щоб перевірити, чи вкладаються витрати у встановлений ліміт бюджету — викликай `analyze_expense_budget`.
5. Якщо отримано помилку валідації аргументів від інструменту — уважно прочитай підказку, виправ параметри та повтори виклик.

ОБМЕЖЕННЯ:
- Не вигадуй фінансові цифри самостійно — спирайся лише на дані з інструментів.
- Відповідай українською мовою, чітко та структуровано.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. Pydantic-схема структурованої фінальної відповіді
# ══════════════════════════════════════════════════════════════════════════════
class AgentStructuredResponse(BaseModel):
    """Структурований формат фінальної відповіді фінансового агента."""

    answer: str = Field(
        description="Повна або часткова відповідь користувачу українською мовою з точними цифрами."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Оцінка впевненості у відповіді від 0.0 до 1.0.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Список назв інструментів (tools), використаних під час роботи.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Стан графа (State)
# ══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    """Стан ReAct-агента, що передається між вузлами графа."""

    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    stop_reason: Optional[str]
    structured_response: Optional[dict]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Ініціалізація LLM через OpenRouter
# ══════════════════════════════════════════════════════════════════════════════
def get_llm(model_name: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Створює клієнт ChatOpenAI, підключений до OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Не знайдено OPENROUTER_API_KEY. Додайте ключ у файл .env."
        )

    if "gemini-3" in model_name:
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            timeout=30,
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        temperature=0.0,
        timeout=30,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Побудова захищеного LangGraph з логуванням траєкторії
# ══════════════════════════════════════════════════════════════════════════════
def build_react_graph(
    safety_guard: Optional[SafetyGuard] = None,
    trajectory_logger: Optional[TrajectoryLogger] = None,
    model_name: str = DEFAULT_MODEL,
    verbose: bool = True,
):
    """Збирає та компілює LangGraph StateGraph із захисними механізмами та логуванням."""
    guard = safety_guard or SafetyGuard()
    logger = trajectory_logger or TrajectoryLogger()

    llm = get_llm(model_name=model_name)
    llm_with_tools = llm.bind_tools(FINANCIAL_TOOLS)
    structured_llm = llm.with_structured_output(
        AgentStructuredResponse, method="function_calling"
    )
    tools_by_name = {t.name: t for t in FINANCIAL_TOOLS}

    # ── Вузол 1: agent (LLM з перевіркою max_steps та timeout) ──────
    def agent_node(state: AgentState) -> dict:
        node_start = time.monotonic()
        current_step = state.get("step_count", 0)
        last_input = state["messages"][-1].content if state["messages"] else ""

        # Перевірка захисних лімітів перед викликом LLM
        should_stop, reason = guard.check_before_step(current_step)
        if should_stop:
            partial_msg = (
                f"⚠️ Виконання зупинено захисним механізмом: {reason}\n"
                f"Частковий результат: виконано {current_step} крок(ів) аналізу. "
                f"Наявні спостереження збережено в історії."
            )
            if verbose:
                print(f"\n[Node: agent] ЗУПИНКА: {reason}")

            duration_ms = int((time.monotonic() - node_start) * 1000)
            logger.log_step(
                step_number=len(logger.steps) + 1,
                node_name="agent",
                input_data=last_input,
                output_data=partial_msg,
                tool_calls=[],
                duration_ms=duration_ms,
            )
            return {
                "messages": [AIMessage(content=partial_msg)],
                "step_count": current_step,
                "stop_reason": reason,
            }

        next_step = current_step + 1
        if verbose:
            print(f"\n[Node: agent] Крок #{next_step} — аналіз контексту...")

        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        else:
            messages_for_llm = list(messages)

        response = llm_with_tools.invoke(messages_for_llm)
        duration_ms = int((time.monotonic() - node_start) * 1000)

        raw_tool_calls = getattr(response, "tool_calls", None) or []
        logged_calls = [{"name": tc["name"], "args": tc["args"]} for tc in raw_tool_calls]

        if verbose:
            if logged_calls:
                print(f"[Node: agent] Вирішено викликати tools: {logged_calls}")
            else:
                print("[Node: agent] Вирішено сформувати фінальну відповідь.")

        output_preview = (
            f"tool_calls: {logged_calls}" if logged_calls else (response.content or "")
        )
        logger.log_step(
            step_number=len(logger.steps) + 1,
            node_name="agent",
            input_data=last_input,
            output_data=output_preview,
            tool_calls=logged_calls,
            duration_ms=duration_ms,
        )

        return {
            "messages": [response],
            "step_count": next_step,
        }

    # ── Вузол 2: tools (безпечне виконання + LoopDetector + timeout) ─
    def safe_tool_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        stop_reason: Optional[str] = None

        for tool_call in getattr(last_msg, "tool_calls", []):
            tool_start = time.monotonic()
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            call_id = tool_call["id"]

            # Перевірка на зациклення та тайм-аут перед викликом інструменту
            should_stop, reason = guard.check_tool_execution(tool_name, tool_args)
            if should_stop:
                stop_reason = reason
                halt_text = f"⚠️ Зупинка виконання інструментів: {reason}"
                if verbose:
                    print(f"[Node: tools] ЗУПИНКА: {reason}")

                duration_ms = int((time.monotonic() - tool_start) * 1000)
                logger.log_step(
                    step_number=len(logger.steps) + 1,
                    node_name="tools",
                    input_data=f"{tool_name}({tool_args})",
                    output_data=halt_text,
                    tool_calls=[{"name": tool_name, "args": tool_args}],
                    duration_ms=duration_ms,
                )
                tool_messages.append(ToolMessage(content=halt_text, tool_call_id=call_id))
                break

            if verbose:
                print(f"[Node: tools] Виконання '{tool_name}' з аргументами: {tool_args}")

            if tool_name not in tools_by_name:
                observation = (
                    f"Помилка: інструмент '{tool_name}' не існує. "
                    f"Доступні інструменти: {list(tools_by_name.keys())}"
                )
            else:
                try:
                    observation = str(tools_by_name[tool_name].invoke(tool_args))
                except Exception as exc:
                    observation = f"Помилка виклику {tool_name}: {type(exc).__name__}: {exc}"
                    if verbose:
                        print(f"[Node: tools] Перехоплено помилку: {observation}")

            duration_ms = int((time.monotonic() - tool_start) * 1000)
            logger.log_step(
                step_number=len(logger.steps) + 1,
                node_name="tools",
                input_data=f"{tool_name}({tool_args})",
                output_data=observation,
                tool_calls=[{"name": tool_name, "args": tool_args}],
                duration_ms=duration_ms,
            )
            tool_messages.append(ToolMessage(content=observation, tool_call_id=call_id))

        update: dict = {"messages": tool_messages}
        if stop_reason:
            update["stop_reason"] = stop_reason
        return update

    # ── Conditional Edge 1: після agent ────────────────────────────
    def route_after_agent(state: AgentState) -> Literal["tools", "formatter"]:
        """Якщо зупинка не спрацювала і є tool_calls → tools, інакше → formatter."""
        if state.get("stop_reason"):
            return "formatter"
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            return "tools"
        return "formatter"

    # ── Conditional Edge 2: після tools ────────────────────────────
    def route_after_tools(state: AgentState) -> Literal["agent", "formatter"]:
        """Якщо в tools виявлено зациклення або timeout → одразу у formatter."""
        if state.get("stop_reason"):
            return "formatter"
        return "agent"

    # ── Вузол 3: formatter (Structured Output у Pydantic) ──────────
    def formatter_node(state: AgentState) -> dict:
        node_start = time.monotonic()
        if verbose:
            print("[Node: formatter] Формування структурованої відповіді...")

        used_tools: list[str] = []
        observations: list[str] = []
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc["name"] not in used_tools:
                        used_tools.append(tc["name"])
            elif isinstance(msg, ToolMessage):
                observations.append(msg.content)

        stop_reason = state.get("stop_reason")
        last_text = state["messages"][-1].content or ""

        # Якщо спрацював захисний ліміт, формуємо часткову відповідь без зайвого виклику LLM
        if stop_reason:
            partial_summary = (
                f"Виконання перервано захисним механізмом ({stop_reason}).\n\n"
                f"Зібрані проміжні дані ({len(observations)} спостережень):\n"
                + ("\n---\n".join(observations) if observations else last_text)
            )
            structured = AgentStructuredResponse(
                answer=partial_summary,
                confidence=0.4,
                sources=used_tools,
            )
        else:
            prompt = (
                "На основі фінальної відповіді асистента та використаних інструментів "
                "сформуй структурований JSON-об'єкт.\n"
                f"Використані інструменти: {used_tools}\n"
                f"Текст відповіді асистента:\n{last_text}"
            )
            try:
                structured = structured_llm.invoke(prompt)
                if not structured.sources and used_tools:
                    structured.sources = used_tools
            except Exception:
                structured = AgentStructuredResponse(
                    answer=last_text,
                    confidence=0.9 if used_tools else 0.7,
                    sources=used_tools,
                )

        result_dict = structured.model_dump()
        duration_ms = int((time.monotonic() - node_start) * 1000)
        logger.log_step(
            step_number=len(logger.steps) + 1,
            node_name="formatter",
            input_data=last_text,
            output_data=result_dict["answer"],
            tool_calls=[],
            duration_ms=duration_ms,
        )
        return {"structured_response": result_dict}

    # ── Збирання графа ─────────────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", safe_tool_node)
    workflow.add_node("formatter", formatter_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "formatter": "formatter"},
    )
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "formatter": "formatter"},
    )
    workflow.add_edge("formatter", END)

    return workflow.compile()


def run_agent(
    query: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_repeats: int = DEFAULT_MAX_REPEATS,
    model_name: str = DEFAULT_MODEL,
    verbose: bool = True,
    save_trajectory_path: Optional[str] = None,
) -> dict:
    """Запускає повний цикл ReAct-агента із захистом та логуванням траєкторії."""
    guard = SafetyGuard(
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        max_repeats=max_repeats,
    )
    logger = TrajectoryLogger(query=query)
    app = build_react_graph(
        safety_guard=guard,
        trajectory_logger=logger,
        model_name=model_name,
        verbose=verbose,
    )

    start_mono = time.monotonic()
    final_state = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "step_count": 0,
            "stop_reason": None,
            "structured_response": None,
        }
    )
    elapsed_ms = int((time.monotonic() - start_mono) * 1000)

    stop_reason = final_state.get("stop_reason")
    status = "stopped_by_guard" if stop_reason else "completed"

    if save_trajectory_path:
        logger.save(filepath=save_trajectory_path, status=status, stop_reason=stop_reason)

    # Збираємо перелік фактично виконаних інструментів 
    called_tools: list[str] = []
    for step in logger.steps:
        if step.get("node_name") == "tools":
            for tc in step.get("tool_calls", []):
                called_tools.append(tc["name"])

    return {
        "query": query,
        "status": status,
        "stop_reason": stop_reason,
        "step_count": final_state["step_count"],
        "tool_calls": called_tools,
        "duration_ms": elapsed_ms,
        "structured_response": final_state["structured_response"],
        "trajectory": logger.to_dict(status=status, stop_reason=stop_reason),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Демонстрація захисних механізмів та збереження trajectory.json
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from safety import LoopDetector

    print("=" * 70)
    print("1. ШТАТНИЙ ЗАПУСК З ЛОГУВАННЯМ ТРАЄКТОРІЇ У trajectory.json")
    print("=" * 70)
    normal_res = run_agent(
        query="Які були доходи та витрати у 2026-10? Порахуй чистий прибуток і маржинальність при податку 5%.",
        max_steps=10,
        timeout_seconds=60.0,
        verbose=True,
        save_trajectory_path="trajectory.json",
    )
    print(f"Статус: {normal_res['status']} | Кроків: {normal_res['step_count']} | Час: {normal_res['duration_ms']} мс")

    print("\n" + "=" * 70)
    print("2. ДЕМОНСТРАЦІЯ ЗАХИСТУ max_steps (ліміт = 1 крок на багатокроковий запит)")
    print("=" * 70)
    max_steps_res = run_agent(
        query="Отримай записи за 2026-10, порахуй прибуток при податку 5% і перевір бюджет 180000 грн.",
        max_steps=1,
        timeout_seconds=60.0,
        verbose=True,
    )
    print(f"Статус: {max_steps_res['status']}")
    print(f"Причина зупинки: {max_steps_res['stop_reason']}")
    print(f"Часткова відповідь:\n{max_steps_res['structured_response']['answer'][:350]}...")

    print("\n" + "=" * 70)
    print("3. ДЕМОНСТРАЦІЯ ЗАХИСТУ timeout (ліміт = 0.001 секунди)")
    print("=" * 70)
    timeout_res = run_agent(
        query="Перевір бюджет витрат за 2026-10 при ліміті 180000 грн.",
        max_steps=10,
        timeout_seconds=0.001,
        verbose=True,
    )
    print(f"Статус: {timeout_res['status']}")
    print(f"Причина зупинки: {timeout_res['stop_reason']}")

    print("\n" + "=" * 70)
    print("4. ДЕМОНСТРАЦІЯ ДЕТЕКЦІЇ ЗАЦИКЛЕННЯ (LoopDetector: 3 однакові виклики)")
    print("=" * 70)
    detector = LoopDetector(max_repeats=3)
    for attempt in range(1, 4):
        is_loop = detector.check("get_financial_records", {"period": "2026-10", "record_type": "all"})
        print(f"Спроба #{attempt}: get_financial_records(period='2026-10') -> loop_detected={is_loop}")
