"""
Модуль фінансових інструментів (tools) для ReAct-агента з Pydantic v2 валідацією.
Домен: контроль доходів, витрат, розрахунок прибутку та бюджетний аудит.
"""

import re
from typing import Literal, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# Дозволені категорії фінансових операцій
ALLOWED_CATEGORIES = {
    "sales",        # Продажі / виручка від клієнтів
    "consulting",   # Консалтинг та послуги
    "investments",  # Інвестиційний дохід
    "salaries",     # Зарплати та підрядники
    "software",     # ПЗ, хмарні сервіси, API
    "marketing",    # Маркетинг та реклама
    "office",       # Офісні та операційні витрати
    "taxes",        # Податки та збори
}

# Імітація бази даних фінансових транзакцій (у гривнях, UAH)
FINANCIAL_LEDGER: list[dict] = [
    {"id": "TX-001", "period": "2026-09", "type": "income", "category": "sales", "amount": 240000.0, "description": "Оплата від клієнта Alpha"},
    {"id": "TX-002", "period": "2026-09", "type": "expense", "category": "salaries", "amount": 110000.0, "description": "Виплати команді розробки"},
    {"id": "TX-003", "period": "2026-09", "type": "expense", "category": "software", "amount": 18500.0, "description": "Хмарна інфраструктура AWS та LLM API"},
    {"id": "TX-004", "period": "2026-10", "type": "income", "category": "sales", "amount": 310000.0, "description": "Контракт з клієнтом Beta"},
    {"id": "TX-005", "period": "2026-10", "type": "income", "category": "consulting", "amount": 45000.0, "description": "Аудит AI-процесів"},
    {"id": "TX-006", "period": "2026-10", "type": "expense", "category": "salaries", "amount": 135000.0, "description": "Зарплатний фонд за жовтень"},
    {"id": "TX-007", "period": "2026-10", "type": "expense", "category": "marketing", "amount": 42000.0, "description": "Таргетована реклама B2B"},
    {"id": "TX-008", "period": "2026-10", "type": "expense", "category": "software", "amount": 23000.0, "description": "Підписки SaaS та OpenRouter"},
    {"id": "TX-009", "period": "2026-11", "type": "income", "category": "sales", "amount": 280000.0, "description": "Передплата за листопад"},
    {"id": "TX-010", "period": "2026-11", "type": "expense", "category": "salaries", "amount": 140000.0, "description": "Зарплати за листопад"},
]


def _validate_period_format(value: str) -> str:
    """Допоміжна функція перевірки формату періоду YYYY-MM."""
    if not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", value):
        raise ValueError(
            f"Некоректний формат періоду '{value}'. Очікується YYYY-MM (наприклад, '2026-10')."
        )
    year = int(value.split("-")[0])
    if not (2020 <= year <= 2030):
        raise ValueError("Рік періоду має бути в діапазоні 2020–2030.")
    return value


# ══════════════════════════════════════════════════════════════════════════════
# 1. Інструмент отримання фінансових записів
# ══════════════════════════════════════════════════════════════════════════════
class FinancialRecordsInput(BaseModel):
    """Параметри пошуку фінансових операцій у реєстрі."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    period: str = Field(
        description="Звітний місяць у форматі YYYY-MM (наприклад, '2026-10')."
    )
    record_type: Literal["all", "income", "expense"] = Field(
        default="all",
        description="Тип операцій: 'income' (доходи), 'expense' (витрати) або 'all' (усі).",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Опціональна категорія для фільтрації: sales, consulting, investments, "
            "salaries, software, marketing, office, taxes."
        ),
    )

    @field_validator("period")
    @classmethod
    def check_period(cls, v: str) -> str:
        return _validate_period_format(v)

    @field_validator("category")
    @classmethod
    def check_category(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        normalized = v.lower()
        if normalized not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Невідома категорія '{v}'. Дозволені: {sorted(ALLOWED_CATEGORIES)}."
            )
        return normalized


@tool(args_schema=FinancialRecordsInput)
def get_financial_records(
    period: str,
    record_type: Literal["all", "income", "expense"] = "all",
    category: Optional[str] = None,
) -> str:
    """Отримати список фінансових операцій (доходів та витрат) за вказаний місяць.

    Коли використовувати:
    - Потрібно дізнатися фактичні доходи, витрати або список транзакцій за конкретний місяць.
    - Потрібно зібрати вихідні суми перед розрахунком чистого прибутку чи аналізом бюджету.

    Коли НЕ використовувати:
    - Для математичного обчислення маржі або податків (для цього є calculate_profit_metrics).
    - Для додавання нової транзакції (для цього є record_transaction).

    Приклад виклику:
        get_financial_records(period="2026-10", record_type="expense")

    Args:
        period: Місяць у форматі YYYY-MM (наприклад, '2026-10').
        record_type: Фільтр типу ('all', 'income', 'expense').
        category: Опціональна назва категорії.

    Returns:
        Структурований звіт зі списком знайдених операцій та підсумковими сумами у UAH.
    """
    filtered = [
        tx for tx in FINANCIAL_LEDGER
        if tx["period"] == period
        and (record_type == "all" or tx["type"] == record_type)
        and (category is None or tx["category"] == category)
    ]

    if not filtered:
        return f"За період {period} (тип={record_type}, категорія={category}) транзакцій не знайдено."

    total_income = sum(tx["amount"] for tx in filtered if tx["type"] == "income")
    total_expense = sum(tx["amount"] for tx in filtered if tx["type"] == "expense")

    lines = [f"Фінансові записи за {period} (знайдено {len(filtered)}):"]
    for tx in filtered:
        sign = "+" if tx["type"] == "income" else "-"
        lines.append(
            f"- [{tx['id']}] {tx['type'].upper()} | {tx['category']}: "
            f"{sign}{tx['amount']:.2f} UAH ({tx['description']})"
        )
    lines.append(f"Разом доходи: {total_income:.2f} UAH | Разом витрати: {total_expense:.2f} UAH")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Інструмент розрахунку прибутку та рентабельності
# ══════════════════════════════════════════════════════════════════════════════
class ProfitMetricsInput(BaseModel):
    """Параметри для обчислення показників прибутку та маржинальності."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    total_revenue: float = Field(
        description="Загальний дохід (виручка) у UAH. Має бути >= 0."
    )
    total_expenses: float = Field(
        description="Загальні операційні витрати у UAH. Має бути >= 0."
    )
    tax_rate_percent: float = Field(
        default=5.0,
        description="Ставка податку у відсотках від доходу або прибутку (0–100%, за замовчуванням 5%).",
    )

    @field_validator("total_revenue", "total_expenses")
    @classmethod
    def non_negative_amount(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Сума доходу або витрат не може бути від'ємною.")
        if v > 1_000_000_000:
            raise ValueError("Сума перевищує максимально допустимий ліміт (1 млрд UAH).")
        return round(v, 2)

    @field_validator("tax_rate_percent")
    @classmethod
    def valid_tax_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 100.0):
            raise ValueError("Податкова ставка має бути в межах від 0 до 100%.")
        return round(v, 2)


@tool(args_schema=ProfitMetricsInput)
def calculate_profit_metrics(
    total_revenue: float,
    total_expenses: float,
    tax_rate_percent: float = 5.0,
) -> str:
    """Обчислити операційний прибуток, податки, чистий прибуток та рентабельність (маржу).

    Коли використовувати:
    - Коли відомі суми доходів і витрат та потрібно точно порахувати прибуток, податок і маржу у %.
    - Після отримання даних через get_financial_records для фінансового підсумку.

    Коли НЕ використовувати:
    - Якщо ви ще не знаєте суми доходів і витрат за період (спершу викличте get_financial_records).

    Приклад виклику:
        calculate_profit_metrics(total_revenue=355000, total_expenses=200000, tax_rate_percent=5.0)

    Args:
        total_revenue: Загальний дохід у UAH (>= 0).
        total_expenses: Загальні витрати у UAH (>= 0).
        tax_rate_percent: Ставка податку у відсотках (0..100).

    Returns:
        Деталізований фінансовий розрахунок прибутку, податків та маржинальності.
    """
    operating_profit = total_revenue - total_expenses
    # Податок нараховується на додатний операційний прибуток (або 0, якщо збиток)
    tax_amount = round(max(operating_profit, 0.0) * (tax_rate_percent / 100.0), 2)
    net_profit = round(operating_profit - tax_amount, 2)
    net_margin = round((net_profit / total_revenue * 100.0), 2) if total_revenue > 0 else 0.0
    expense_ratio = round((total_expenses / total_revenue * 100.0), 2) if total_revenue > 0 else 0.0

    status = "ПРИБУТКОВИЙ" if net_profit > 0 else ("ЗБИТКОВИЙ" if net_profit < 0 else "ТОЧКА БЕЗЗБИТКОВОСТІ")

    return (
        f"Фінансовий розрахунок ({status}):\n"
        f"- Валовий дохід: {total_revenue:.2f} UAH\n"
        f"- Операційні витрати: {total_expenses:.2f} UAH ({expense_ratio}% від доходу)\n"
        f"- Операційний прибуток (до податків): {operating_profit:.2f} UAH\n"
        f"- Податок ({tax_rate_percent}%): {tax_amount:.2f} UAH\n"
        f"- Чистий прибуток: {net_profit:.2f} UAH\n"
        f"- Чиста маржинальність (Net Margin): {net_margin}%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Інструмент додавання нової транзакції
# ══════════════════════════════════════════════════════════════════════════════
class RecordTransactionInput(BaseModel):
    """Параметри для реєстрації нової фінансової операції."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    transaction_type: Literal["income", "expense"] = Field(
        description="Тип операції: 'income' (надходження коштів) або 'expense' (витрата)."
    )
    category: str = Field(
        description=(
            "Категорія операції: sales, consulting, investments, "
            "salaries, software, marketing, office або taxes."
        )
    )
    amount: float = Field(
        description="Сума операції у UAH. Повинна бути строго більшою за 0."
    )
    period: str = Field(
        description="Місяць проведення операції у форматі YYYY-MM (наприклад, '2026-11')."
    )
    description: str = Field(
        description="Короткий опис призначення платежу (мінімум 3 символи)."
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        normalized = v.lower()
        if normalized not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Неприпустима категорія '{v}'. Дозволені: {sorted(ALLOWED_CATEGORIES)}."
            )
        return normalized

    @field_validator("amount")
    @classmethod
    def positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Сума транзакції має бути строго більшою за 0.")
        return round(v, 2)

    @field_validator("period")
    @classmethod
    def check_period(cls, v: str) -> str:
        return _validate_period_format(v)

    @field_validator("description")
    @classmethod
    def valid_description(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError("Опис транзакції повинен містити щонайменше 3 символи.")
        return v


@tool(args_schema=RecordTransactionInput)
def record_transaction(
    transaction_type: Literal["income", "expense"],
    category: str,
    amount: float,
    period: str,
    description: str,
) -> str:
    """Зареєструвати нову фінансову транзакцію (дохід або витрату) у реєстрі.

    Коли використовувати:
    - Користувач просить додати, записати або зафіксувати новий дохід чи витрату.

    Коли НЕ використовувати:
    - Для перегляду існуючих записів або розрахунку аналітики без додавання даних.

    Приклад виклику:
        record_transaction(
            transaction_type="expense",
            category="software",
            amount=12500,
            period="2026-11",
            description="Оплата ліцензій GitHub Copilot"
        )

    Args:
        transaction_type: 'income' або 'expense'.
        category: Дозволена фінансова категорія.
        amount: Додатна сума у UAH (> 0).
        period: Період YYYY-MM.
        description: Призначення платежу.

    Returns:
        Підтвердження запису з присвоєним ID транзакції.
    """
    new_id = f"TX-{len(FINANCIAL_LEDGER) + 1:03d}"
    entry = {
        "id": new_id,
        "period": period,
        "type": transaction_type,
        "category": category,
        "amount": amount,
        "description": description,
    }
    FINANCIAL_LEDGER.append(entry)
    return (
        f"Транзакцію успішно збережено: [{new_id}] {transaction_type.upper()} | "
        f"категорія={category} | сума={amount:.2f} UAH | період={period} | опис='{description}'"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Інструмент контролю бюджету витрат
# ══════════════════════════════════════════════════════════════════════════════
class BudgetAnalysisInput(BaseModel):
    """Параметри для перевірки дотримання ліміту витрат."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    period: str = Field(
        description="Звітний місяць у форматі YYYY-MM (наприклад, '2026-10')."
    )
    budget_limit_uah: float = Field(
        description="Затверджений ліміт витрат у UAH. Має бути > 0."
    )
    category: Optional[str] = Field(
        default=None,
        description="Опціональна категорія витрат для перевірки (якщо не вказано — перевіряються всі витрати).",
    )

    @field_validator("period")
    @classmethod
    def check_period(cls, v: str) -> str:
        return _validate_period_format(v)

    @field_validator("budget_limit_uah")
    @classmethod
    def positive_limit(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Ліміт бюджету має бути строго більшим за 0 UAH.")
        return round(v, 2)

    @field_validator("category")
    @classmethod
    def check_category(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        normalized = v.lower()
        if normalized not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Невідома категорія '{v}'. Дозволені: {sorted(ALLOWED_CATEGORIES)}."
            )
        return normalized


@tool(args_schema=BudgetAnalysisInput)
def analyze_expense_budget(
    period: str,
    budget_limit_uah: float,
    category: Optional[str] = None,
) -> str:
    """Перевірити фактичні витрати за період проти встановленого ліміту бюджету.

    Коли використовувати:
    - Потрібно оцінити, чи вкладаються витрати (загальні або по конкретній категорії) у бюджет.
    - Потрібно виявити перевитрату (overspend) або залишок бюджету.

    Коли НЕ використовувати:
    - Для розрахунку доходів або чистого прибутку (використовуйте calculate_profit_metrics).

    Приклад виклику:
        analyze_expense_budget(period="2026-10", budget_limit_uah=180000)

    Args:
        period: Місяць у форматі YYYY-MM.
        budget_limit_uah: Плановий ліміт витрат у UAH (> 0).
        category: Опціональна категорія витрат.

    Returns:
        Звіт про виконання бюджету: факт, ліміт, відхилення та відсоток використання.
    """
    expenses = [
        tx for tx in FINANCIAL_LEDGER
        if tx["period"] == period
        and tx["type"] == "expense"
        and (category is None or tx["category"] == category)
    ]
    actual_spent = round(sum(tx["amount"] for tx in expenses), 2)
    diff = round(budget_limit_uah - actual_spent, 2)
    utilization = round((actual_spent / budget_limit_uah) * 100.0, 2)
    scope = f"категорія '{category}'" if category else "усі витрати"

    if diff >= 0:
        verdict = f"У МЕЖАХ БЮДЖЕТУ (залишок: {diff:.2f} UAH)"
    else:
        verdict = f"ПЕРЕВИТРАТА БЮДЖЕТУ на {abs(diff):.2f} UAH!"

    return (
        f"Бюджетний аудит за {period} ({scope}):\n"
        f"- Ліміт бюджету: {budget_limit_uah:.2f} UAH\n"
        f"- Фактичні витрати: {actual_spent:.2f} UAH\n"
        f"- Використання бюджету: {utilization}%\n"
        f"- Статус: {verdict}"
    )


# Список усіх інструментів для передачі в LangGraph-агент
FINANCIAL_TOOLS = [
    get_financial_records,
    calculate_profit_metrics,
    record_transaction,
    analyze_expense_budget,
]


# ══════════════════════════════════════════════════════════════════════════════
# Unit-тести інструментів та Pydantic-валідації
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== 1. Тестування Pydantic-валідації (очікувані помилки) ===")

    invalid_cases = [
        ("Невірний формат дати", lambda: FinancialRecordsInput(period="жовтень 2026")),
        ("Від'ємний дохід", lambda: ProfitMetricsInput(total_revenue=-5000, total_expenses=1000)),
        ("Податок > 100%", lambda: ProfitMetricsInput(total_revenue=10000, total_expenses=2000, tax_rate_percent=120)),
        ("Нульова транзакція", lambda: RecordTransactionInput(
            transaction_type="expense", category="office", amount=0, period="2026-11", description="Канцтовари"
        )),
        ("Неіснуюча категорія", lambda: BudgetAnalysisInput(
            period="2026-10", budget_limit_uah=50000, category="crypto"
        )),
        ("Зайве поле (extra='forbid')", lambda: BudgetAnalysisInput(
            period="2026-10", budget_limit_uah=50000, currency="USD"
        )),
    ]

    for title, fn in invalid_cases:
        try:
            fn()
            print(f"[FAIL] {title}: валідація не спрацювала!")
        except ValidationError as e:
            first_err = e.errors()[0]["msg"]
            print(f"[OK] {title} -> ValidationError: {first_err}")

    print("\n=== 2. Тестування успішних викликів інструментів ===")

    res1 = get_financial_records.invoke({"period": "2026-10"})
    print(f"\n--- get_financial_records ---\n{res1}")

    res2 = calculate_profit_metrics.invoke({
        "total_revenue": 355000.0,
        "total_expenses": 200000.0,
        "tax_rate_percent": 5.0,
    })
    print(f"\n--- calculate_profit_metrics ---\n{res2}")

    res3 = record_transaction.invoke({
        "transaction_type": "expense",
        "category": "software",
        "amount": 15000.0,
        "period": "2026-11",
        "description": "Оплата серверів та моніторингу",
    })
    print(f"\n--- record_transaction ---\n{res3}")

    res4 = analyze_expense_budget.invoke({
        "period": "2026-10",
        "budget_limit_uah": 180000.0,
    })
    print(f"\n--- analyze_expense_budget ---\n{res4}")
