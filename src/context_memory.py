"""
insightx/src/context_memory.py
───────────────────────────────
Manages conversation context for InsightX.
Works directly with AnalyticalQuery from nlp_engine.py.
Enables follow-up queries like "What about Karnataka?" to inherit
the previous metric, group_by, and intent automatically.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Optional, Any
import copy

# TYPE_CHECKING guard avoids circular imports at runtime
if TYPE_CHECKING:
    from src.nlp_engine import AnalyticalQuery


def _query_summary(ctx: "AnalyticalQuery") -> str:
    """Build a one-line summary string from any AnalyticalQuery object."""
    parts = [f"metric={ctx.metric}", f"intent={ctx.intent}"]
    if getattr(ctx, "group_by", None):
        parts.append(f"group_by={ctx.group_by}")
    if getattr(ctx, "filters", None):
        parts.append(f"filters={ctx.filters}")
    if getattr(ctx, "compare", None):
        parts.append(f"compare={ctx.compare}")
    return " | ".join(parts)


class ContextMemory:
    """
    Sliding window of past AnalyticalQuery objects.
    Enables follow-up context inheritance across turns.
    """

    def __init__(self, max_history: int = 10):
        self._history: List[Any] = []   # List[AnalyticalQuery]
        self._max = max_history

    # ── Public API ────────────────────────────────────────────────────────

    def push(self, ctx: "AnalyticalQuery") -> "AnalyticalQuery":
        """
        Store a query. If ctx.followup=True, merges filters/compare
        from the previous query so context carries forward.
        Returns the stored (possibly merged) query.
        """
        if getattr(ctx, "followup", False) and self._history:
            resolved = self._merge(self._history[-1], ctx)
        else:
            resolved = ctx

        self._history.append(resolved)
        if len(self._history) > self._max:
            self._history.pop(0)

        return resolved

    def last(self) -> Optional[Any]:
        """Return the most recent stored query."""
        return self._history[-1] if self._history else None

    def clear(self):
        self._history.clear()

    def to_prompt_context(self, n: int = 3) -> str:
        """
        Return last n queries as a plain-text string to inject into
        parse_query() as context_hint, so follow-ups are detected correctly.
        """
        recent = self._history[-n:]
        if not recent:
            return ""

        lines = ["CONVERSATION HISTORY (most recent last):"]
        for i, ctx in enumerate(recent, 1):
            lines.append(f"  [{i}] User: \"{ctx.raw_query}\"")
            lines.append(f"       → {_query_summary(ctx)}")
        lines.append("")
        lines.append("If the next query is a follow-up, inherit metric/group_by/intent from above.")
        return "\n".join(lines)

    def get_history_display(self) -> List[Dict]:
        """Return history as list of dicts for the sidebar trail UI."""
        return [
            {
                "query":    ctx.raw_query,
                "metric":   ctx.metric,
                "intent":   ctx.intent,
                "group_by": getattr(ctx, "group_by", None) or "—",
                "filters":  getattr(ctx, "filters", {}),
                "followup": getattr(ctx, "followup", False),
            }
            for ctx in reversed(self._history)
        ]

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _merge(prev: "AnalyticalQuery", followup: "AnalyticalQuery") -> "AnalyticalQuery":
        """
        Merge a follow-up onto the previous query.
        - PRESERVE: intent, metric, group_by (unless followup overrides them)
        - MERGE:    filters (additive)
        - REPLACE:  compare, time_window, top_n (if followup sets them)
        """
        merged = copy.deepcopy(prev)
        merged.raw_query = followup.raw_query
        merged.followup  = True

        # Only override if the follow-up detected something non-default
        if followup.intent not in ("single", "summary"):
            merged.intent = followup.intent
        if followup.metric != "count":
            merged.metric = followup.metric
        if getattr(followup, "group_by", None) is not None:
            merged.group_by = followup.group_by

        # Merge filters additively
        if getattr(followup, "filters", None):
            merged.filters.update(followup.filters)

        # Replace compare/time_window/top_n if provided
        if getattr(followup, "compare", None):
            merged.compare = followup.compare
        if getattr(followup, "time_window", None) is not None:
            merged.time_window = followup.time_window
        if getattr(followup, "top_n", 10) != 10:
            merged.top_n = followup.top_n

        return merged


# kept for backwards compat — not used internally anymore
def parse_llm_response(raw_json: str, user_query: str):
    """Legacy stub — no longer needed since we use rule-based nlp_engine."""
    raise NotImplementedError(
        "parse_llm_response is not used. "
        "Use nlp_engine.parse_query() instead."
    )
