"""uptake-probe: a judge-free benchmark for tool-result reliance in LLM agents.

The package plants un-memorizable nonce values into tool results, so whether an
agent relied on a result is scored by exact match. It contains the deterministic
generator, the scorer (exact match plus an abstention detector), and the agent
harness used in "Copy First, Check the Field Later" (Findings of EACL 2027).

Identifiers predate the paper's vocabulary: "confab" is the paper's mis-grounding
and "uptake" its utilization (see README.md).
"""

from .schema import (
    DeliveryMode,
    DistractorType,
    Instance,
    NonceFact,
    OND,
    ToolResult,
)

__all__ = [
    "DeliveryMode",
    "DistractorType",
    "Instance",
    "NonceFact",
    "OND",
    "ToolResult",
]
