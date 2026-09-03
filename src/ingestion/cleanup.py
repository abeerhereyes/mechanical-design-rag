"""Document-level cleanup for recurring page furniture."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Set


def _normalized(line: str) -> str:
    line = re.sub(r"\d+", "#", line.casefold())
    return re.sub(r"\s+", " ", line).strip(" \t-|")


def recurring_marginal_lines(
    pages: Iterable[str], frequency: float = 0.6, margin_lines: int = 2
) -> Set[str]:
    page_list = list(pages)
    if len(page_list) < 2:
        return set()
    counts: Counter[str] = Counter()
    for text in page_list:
        lines = [line for line in text.splitlines() if line.strip()]
        candidates = lines[:margin_lines] + lines[-margin_lines:]
        counts.update(set(filter(None, map(_normalized, candidates))))
    minimum = max(2, int(len(page_list) * frequency + 0.999))
    return {line for line, count in counts.items() if count >= minimum}


def remove_recurring_margins(
    pages: List[str], frequency: float = 0.6, margin_lines: int = 2
) -> List[str]:
    recurring = recurring_marginal_lines(pages, frequency, margin_lines)
    cleaned = []
    for text in pages:
        lines = text.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        marginal = set(nonempty[:margin_lines] + nonempty[-margin_lines:])
        kept = [
            line
            for i, line in enumerate(lines)
            if i not in marginal or _normalized(line) not in recurring
        ]
        cleaned.append("\n".join(kept).strip())
    return cleaned
