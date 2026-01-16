from __future__ import annotations

import re

from fc.dsl.tokens import NAMES
from fc.util.tags import apply_tag, split_domain_tag


def _replace_field(text: str, key: str, new_val: str) -> str | None:
    pattern = rf"({re.escape(key)}\s*(?:[:=]|is)\s*)([^\n;,.]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    def _repl(match_obj: re.Match[str]) -> str:
        return f"{match_obj.group(1)}{new_val}"

    return re.sub(pattern, _repl, text, count=1, flags=re.IGNORECASE)


def _next_name(name: str) -> str:
    if name in NAMES:
        idx = NAMES.index(name)
        return NAMES[(idx + 1) % len(NAMES)]
    for candidate in NAMES:
        if candidate != name:
            return candidate
    return NAMES[0]


def schema_flips(text: str) -> list[str]:
    tag, base = split_domain_tag(text)
    text = base
    age_match = re.search(r"age\s*(?:[:=]|is)\s*(\d+)", text, flags=re.IGNORECASE)
    name_match = re.search(r"name\s*(?:[:=]|is)\s*([^\n;,.]+)", text, flags=re.IGNORECASE)
    flips = []
    if age_match:
        age = int(age_match.group(1))
        updated = _replace_field(text, "age", str(age + 1))
        if updated:
            flips.append(updated)
    if name_match:
        name = name_match.group(1).strip()
        new_name = _next_name(name)
        updated = _replace_field(text, "name", new_name)
        if updated:
            flips.append(updated)
    return [apply_tag(tag, flip) for flip in flips]


def math_flips(text: str) -> list[str]:
    tag, base = split_domain_tag(text)
    text = base
    nums = re.findall(r"-?\d+", text)
    flips = []
    if nums:
        first = nums[0]
        new_first = str(int(first) + 1)
        flips.append(text.replace(first, new_first, 1))
    return [apply_tag(tag, flip) for flip in flips]


def csp_flips(text: str) -> list[str]:
    tag, base = split_domain_tag(text)
    text = base
    flips = []
    tasks = re.findall(r"([A-Z])\s*=\s*(\d+)", text)
    if tasks:
        task, dur = tasks[0]
        new_dur = str(int(dur) + 1)
        updated = re.sub(rf"{task}\s*=\s*{dur}", f"{task}={new_dur}", text, count=1)
        flips.append(updated)
    cons = re.findall(r"([A-Z])\s*<\s*([A-Z])", text)
    if cons:
        a, b = cons[0]
        if "Constraints:" in text:
            updated = text.replace("Constraints:", f"Constraints: {b}<{a},", 1)
        else:
            updated = text + f" Constraints: {b}<{a}."
        flips.append(updated)
    return [apply_tag(tag, flip) for flip in flips]


def generate_flips(domain: str, text: str) -> list[str]:
    if domain == "schema":
        return schema_flips(text)
    if domain == "math":
        return math_flips(text)
    if domain == "csp":
        return csp_flips(text)
    return []
