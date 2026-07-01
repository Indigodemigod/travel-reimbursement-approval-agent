import re
from pathlib import Path
from typing import Any

POLICY_FILE = Path(__file__).resolve().parent.parent / "policy" / "travel_policy.md"
SECTION_PATTERN = re.compile(r"^## .+$", re.MULTILINE)
WORD_PATTERN = re.compile(r"[a-z0-9]+")

NO_MATCH_MESSAGE = "No relevant policy found."
DEFAULT_MAX_SECTIONS = 3


def _read_policy() -> str:
    return POLICY_FILE.read_text(encoding="utf-8")


def _split_sections(content: str) -> list[tuple[str, str]]:
    matches = list(SECTION_PATTERN.finditer(content))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()
        title = match.group(0).strip()
        sections.append((title, section_text))
    return sections


def _extract_keywords(query: str) -> list[str]:
    return WORD_PATTERN.findall(query.lower())


def _score_section(title: str, body: str, keywords: list[str]) -> int:
    title_lower = title.lower()
    body_lower = body.lower()
    score = 0

    for keyword in keywords:
        if keyword in title_lower:
            score += 3
        if keyword in body_lower:
            score += body_lower.count(keyword)

    return score


def lookup_policy(query: str, max_sections: int = DEFAULT_MAX_SECTIONS) -> dict[str, Any]:
    """Return the most relevant policy sections for a query."""
    keywords = _extract_keywords(query)
    if not keywords:
        return {
            "policy_text": NO_MATCH_MESSAGE,
            "section_titles": [],
        }

    sections = _split_sections(_read_policy())
    scored_sections: list[tuple[int, str, str]] = []

    for title, body in sections:
        score = _score_section(title, body, keywords)
        if score > 0:
            scored_sections.append((score, title, body))

    if not scored_sections:
        return {
            "policy_text": NO_MATCH_MESSAGE,
            "section_titles": [],
        }

    scored_sections.sort(key=lambda item: item[0], reverse=True)
    top_sections = scored_sections[:max_sections]
    section_titles = [title for _, title, _ in top_sections]
    policy_text = "\n\n---\n\n".join(body for _, _, body in top_sections)

    return {
        "policy_text": policy_text,
        "section_titles": section_titles,
    }
