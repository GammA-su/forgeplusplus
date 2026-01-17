from __future__ import annotations

import re

from typing import Tuple

DOMAIN_TAGS = {
    "schema": "[SCHEMA]",
    "math": "[MATH]",
    "csp": "[CSP]",
}

TAG_TO_DOMAIN = {tag: domain for domain, tag in DOMAIN_TAGS.items()}
DOMAIN_TAG_PATTERN = "|".join(re.escape(tag) for tag in DOMAIN_TAGS.values())


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


def domain_from_tag(text: str) -> str | None:
    for tag, domain in TAG_TO_DOMAIN.items():
        if text.startswith(tag):
            return domain
    return None
