"""A mock tool environment built from an Instance.

The agent interacts only through SEARCH/FETCH; the env records which results were
actually observed (fetched), which is what makes the Omission vs Neglect
distinction measurable: a result the agent never fetched cannot have been used.
"""

from __future__ import annotations

import re

from .schema import Instance, ToolResult

_ID = re.compile(r"r\d+")


class ToolEnv:
    def __init__(self, instance: Instance):
        self.instance = instance
        self._by_id: dict[str, ToolResult] = {r.result_id: r for r in instance.results}
        self.observed: set[str] = set()   # result_ids returned by FETCH
        self.search_count = 0
        self.fetch_count = 0

    def search(self, query: str = "") -> str:
        """List every available result id with its snippet (order = position)."""
        self.search_count += 1
        lines = [f"[{r.result_id}] ({r.source}) {r.snippet}" for r in self.instance.results]
        return "\n".join(lines)

    def fetch(self, arg: str) -> str:
        """Return full text for one or more result ids and mark them observed.

        Accepts a single id ("r2") or several ("r2, r3, r4") — batch retrieval is
        a legitimate agent strategy, so we don't penalise it as a syntax error
        (some models, e.g. aya-expanse, batch by default).
        """
        ids = _ID.findall(arg)
        if not ids:
            self.fetch_count += 1
            return f"ERROR: no valid result id in {arg!r}"
        chunks = []
        for rid in ids:
            self.fetch_count += 1
            result = self._by_id.get(rid)
            if result is None:
                chunks.append(f"ERROR: no result with id {rid!r}")
            else:
                self.observed.add(rid)
                chunks.append(f"[{result.result_id}] ({result.source}) {result.full_text}")
        return "\n\n".join(chunks)

    def observed_necessary(self) -> set[str]:
        """Entities whose necessary result was actually fetched."""
        seen = set()
        for r in self.instance.results:
            if r.is_necessary and r.result_id in self.observed and r.fact is not None:
                seen.add(r.fact.entity)
        return seen

    def passive_block(self) -> str:
        """Full text of every result, as shown in passive delivery."""
        return "\n\n".join(
            f"[{r.result_id}] ({r.source}) {r.full_text}" for r in self.instance.results
        )
