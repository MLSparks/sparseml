"""sparkrun: lean, verified, crash-safe execution for AI agents on small models.

Answers three questions for every run:
  1. Did it do the right thing?   independent checks, not the model's word
  2. Can it prove it?             an append-only journal of every step and its evidence
  3. Can it do better next time?  cost per verified task, model escalation, variant learning
"""
from . import compute, verify
from .evolve import Variants
from .journal import Journal, stable_hash
from .memory import TaskState
from .models import Model, OpenAICompatible, PermanentError, ScriptedModel, Usage
from .report import format_report, summarize
from .runner import Agent, ParseError, Result, approve_pending, parse_action
from .team import Role, Team, TeamResult
from .tools import Policy, Tool, TransientError, tool
from .verify import Evidence, Verdict

__version__ = "0.1.0"
__all__ = [
    "Agent", "Evidence", "Role", "Team", "TeamResult", "compute", "Journal", "Model", "OpenAICompatible", "ParseError", "PermanentError",
    "Policy", "Result", "ScriptedModel", "TaskState", "Tool", "TransientError", "Usage", "Variants",
    "Verdict", "approve_pending", "format_report", "parse_action", "stable_hash", "summarize", "tool",
    "verify",
]
