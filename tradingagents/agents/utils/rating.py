"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import re
from typing import Tuple


# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: Tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Chinese rating words → canonical English rating.  The Portfolio Manager
# renders decisions in the language configured by ``output_language``, and
# the "Rating: X" label regex only matches English — so when the PM writes
# ``**评级：减持 (Underweight)**`` the structured-label pass misses it and
# the word-scan pass must handle it.
_CN_RATING_MAP: dict[str, str] = {
    "买入": "Buy",
    "增持": "Overweight",
    "持有": "Hold",
    "减持": "Underweight",
    "卖出": "Sell",
    # Parenthesised English fallback: "(underweight)" after stripping
    # markdown bold markers.
    "underweight": "Underweight",
    "overweight": "Overweight",
}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)

# Matches the common Chinese pattern "评级：减持" / "评级：**减持**"
_CN_RATING_LABEL_RE = re.compile(r"评级[：:]\s*\**(\w+)\**", re.IGNORECASE)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Three-pass strategy:
    1. Look for an explicit "Rating: X" label (English, tolerant of markdown).
    2. Look for Chinese "评级：X" label.
    3. Fall back to the first 5-tier rating word found anywhere in the text
       (English or Chinese).

    Returns a Title-cased rating string, or ``default`` if no rating word appears.
    """
    # Pass 1: English "Rating: X" label
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    # Pass 2: Chinese "评级：X" label
    for line in text.splitlines():
        m = _CN_RATING_LABEL_RE.search(line)
        if m:
            cn_word = m.group(1)
            if cn_word in _CN_RATING_MAP:
                return _CN_RATING_MAP[cn_word]

    # Pass 3: word-by-word scan (English + Chinese)
    for line in text.splitlines():
        for word in line.lower().split():
            # Strip markdown formatting AND parentheses: "(underweight)**" → "underweight"
            clean = word.strip("*:.,()（）")
            if clean in _RATING_SET:
                return clean.capitalize()
            if clean in _CN_RATING_MAP:
                return _CN_RATING_MAP[clean]

    # Pass 4: Chinese substring scan — Chinese prose omits word separators,
    # so "建议减持该标的" won't surface "减持" in a word-split pass.
    # Scan lines in reverse; within a line, pick the *rightmost* Chinese
    # rating word (the conclusion typically appears after the alternatives).
    for line in reversed(text.splitlines()):
        best_pos = -1
        best_rating = None
        for cn_word, eng_rating in _CN_RATING_MAP.items():
            pos = line.rfind(cn_word)
            if pos > best_pos:
                best_pos = pos
                best_rating = eng_rating
        if best_rating is not None:
            return best_rating

    return default
