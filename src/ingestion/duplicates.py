"""Content hashing and deterministic near-duplicate detection."""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional, Set, Tuple

from .schemas import ContentUnit


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _shingles(text: str, width: int = 5) -> Set[Tuple[str, ...]]:
    words = re.findall(r"\w+", normalize_text(text))
    if len(words) < width:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + width]) for i in range(len(words) - width + 1)}


def similarity(left: str, right: str) -> float:
    a, b = _shingles(left), _shingles(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def nearest_duplicate(
    text: str,
    candidates: Iterable[ContentUnit],
    threshold: float = 0.88,
) -> Optional[str]:
    if len(normalize_text(text)) < 80:
        return None
    digest = content_hash(text)
    best_id, best_score = None, threshold
    for candidate in candidates:
        if candidate.content_hash == digest:
            return candidate.unit_id
        score = similarity(text, candidate.text)
        if score >= best_score:
            best_id, best_score = candidate.unit_id, score
    return best_id
