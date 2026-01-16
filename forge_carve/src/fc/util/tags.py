from __future__ import annotations

from typing import Tuple

DOMAIN_TAGS = {
    "schema": "[SCHEMA]",
    "math": "[MATH]",
    "csp": "[CSP]",
}


def split_domain_tag(text: str) -> Tuple[str | None, str]:
    for tag in DOMAIN_TAGS.values():
        if text.startswith(tag):
            return tag, text[len(tag) :].lstrip()
    return None, text


def apply_tag(tag: str | None, text: str) -> str:
    if not tag:
        return text
    if text.startswith(tag):
        return text
    if text:
        return f"{tag} {text}"
    return tag


def apply_domain_tag(domain: str, text: str) -> str:
    return apply_tag(DOMAIN_TAGS.get(domain), text)
