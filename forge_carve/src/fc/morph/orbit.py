from __future__ import annotations

import re
from typing import Iterable

from fc.dsl.tokens import CITIES, NAMES
from fc.util.tags import apply_tag, split_domain_tag


def _find_field(text: str, key: str) -> str | None:
    pattern = rf"{re.escape(key)}\s*(?:[:=]|is)\s*([^\n;,.]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _parse_schema(text: str) -> dict[str, str | int]:
    name = _find_field(text, "name") or NAMES[0]
    age_val = _find_field(text, "age")
    city = _find_field(text, "city") or CITIES[0]
    age = 30
    if age_val:
        digits = re.findall(r"-?\d+", age_val)
        if digits:
            age = int(digits[0])
    return {"name": name, "age": age, "city": city}


def schema_orbits(text: str) -> list[str]:
    tag, base = split_domain_tag(text)
    fields = _parse_schema(base)
    templates = [
        "Record: name={name}; age={age}; city={city}.",
        "Profile: city={city}, name={name}, age={age}.",
        "name={name}; city={city}; age={age}.",
        "Name is {name}; Age is {age}; City is {city}.",
        "- name: {name}\n- age: {age}\n- city: {city}",
        "- age: {age}\n- city: {city}\n- name: {name}",
        "Details:\n- name: {name}\n- age: {age}\n- city: {city}\n- note: ignore",
        "Record: name={name}; age={age}; city={city}.\nNote: ignore this line.",
    ]
    orbits = [t.format(**fields) for t in templates]
    return [apply_tag(tag, orbit) for orbit in orbits]


def _parse_math(text: str) -> tuple[int, int, str]:
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    if len(nums) < 2:
        nums = [1, 1]
    if re.search(r"\+|plus|add", text, flags=re.IGNORECASE):
        op = "plus"
    elif re.search(r"-|minus|subtract", text, flags=re.IGNORECASE):
        op = "minus"
    elif re.search(r"\*|times|multiply", text, flags=re.IGNORECASE):
        op = "times"
    else:
        op = "divide"
    return nums[0], nums[1], op


def math_orbits(text: str) -> list[str]:
    tag, base = split_domain_tag(text)
    a, b, op = _parse_math(base)
    templates = [
        "Compute: {a} {sym} {b}.",
        "What is {a} {op} {b}?",
        "Calculate {a} {sym} {b} quickly.",
        "First number: {a}. Second number: {b}. Operation: {sym}.",
        "a={a}; b={b}; op={sym}.",
        "Please {op_word} the numbers {a} and {b}.",
        "Compute {a} {sym} {b}. Note: ignore formatting.",
    ]
    sym = {"plus": "+", "minus": "-", "times": "*", "divide": "/"}.get(op, "+")
    op_word = {"plus": "add", "minus": "subtract", "times": "multiply", "divide": "divide"}[op]
    orbits = [t.format(a=a, b=b, op=op, sym=sym, op_word=op_word) for t in templates]
    return [apply_tag(tag, orbit) for orbit in orbits]


def _parse_csp(text: str) -> tuple[dict[str, int], list[tuple[str, str]]]:
    tasks: dict[str, int] = {}
    for name, dur in re.findall(r"([A-Z])\s*=\s*(\d+)", text):
        tasks[name] = int(dur)
    constraints = re.findall(r"([A-Z])\s*<\s*([A-Z])", text)
    return tasks, constraints


def csp_orbits(text: str) -> list[str]:
    tag, base = split_domain_tag(text)
    tasks, constraints = _parse_csp(base)
    task_items = list(tasks.items())
    task_part = ",".join(f"{k}={v}" for k, v in task_items)
    task_part_rev = ",".join(f"{k}={v}" for k, v in reversed(task_items))
    cons_part = ",".join(f"{a}<{b}" for a, b in constraints)
    cons_part_rev = ",".join(f"{a}<{b}" for a, b in reversed(constraints))
    templates = [
        f"Tasks: {task_part}. Constraints: {cons_part}.",
        f"Schedule tasks ({task_part}); precedence: {cons_part}.",
        f"Given tasks {task_part_rev} with constraints {cons_part_rev}.",
        "Tasks:\n- "
        + "\n- ".join(f"{k}={v}" for k, v in task_items)
        + "\nConstraints:\n- "
        + "\n- ".join(f"{a}<{b}" for a, b in constraints),
        f"Tasks: {task_part}. Constraints: {cons_part}. Note: single resource.",
    ]
    return [apply_tag(tag, orbit) for orbit in templates]


def generate_orbits(domain: str, text: str) -> list[str]:
    if domain == "schema":
        return schema_orbits(text)
    if domain == "math":
        return math_orbits(text)
    if domain == "csp":
        return csp_orbits(text)
    return []


def choose_orbits(domain: str, text: str, k: int) -> list[str]:
    orbits = generate_orbits(domain, text)
    return list(orbits[:k])
