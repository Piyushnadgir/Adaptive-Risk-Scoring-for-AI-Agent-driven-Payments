from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionContext:
    """Everything known about one agent action. Filled in as stages run."""
    user_instruction: str
    agent_id: str
    amount: float
    payee: str
    tool_call: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    stage: str
    score: float              # 0 (safe) .. 1 (risky)
    flags: list[str] = field(default_factory=list)
    reason: str = ""
