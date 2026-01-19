from __future__ import annotations

import hashlib
import json
import math
import random
import re
from fractions import Fraction
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from pydantic import BaseModel, Field

from fc.dsl.codec import decode_program, encode_program, program_to_tokens
from fc.dsl.program import Instruction, Program
from fc.dsl.tokens import CITIES, NAMES, TASKS, TokenVocab, build_default_vocab
from fc.morph.flips import generate_flips, _next_name
from fc.morph.orbit import generate_orbits
from fc.morph.equiv import outputs_equivalent
from fc.util.runtime_solve import runtime_solve
from fc.util.jsonl import read_jsonl, write_jsonl
from fc.util.math_expr import eval_math_expression, parse_math_expression_ast
from fc.util.tags import DOMAIN_TAGS, DOMAIN_TAG_PATTERN, apply_domain_tag
import numcanon

_TOKEN_PATTERN = re.compile(fr"{DOMAIN_TAG_PATTERN}|\b\w+\b|[+\-*/]")


class ConstraintSpec(BaseModel):
    id: str
    type: str
    args: dict[str, Any] = Field(default_factory=dict)


class Variant(BaseModel):
    x: str
    y: Any | None = None


class Example(BaseModel):
    id: str
    domain: str
    domain_tag: str = ""
    x: str
    y: Any
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    proof: dict[str, Any]
    proof_tokens_gold: list[str] | None = None
    orbit: list[Variant] = Field(default_factory=list)
    flips: list[Variant] = Field(default_factory=list)


def _tokenize_text(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def _canon_y(value: Any) -> Any:
    return numcanon.canon_json(value)


@dataclass(frozen=True)
class TextVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @staticmethod
    def build(texts: Iterable[str]) -> "TextVocab":
        tokens: set[str] = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        for text in texts:
            for tok in _tokenize_text(text):
                if tok in DOMAIN_TAGS.values():
                    tokens.add(tok)
                else:
                    tokens.add(tok.lower())
        ordered = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + sorted(tokens - {"<PAD>", "<UNK>", "<BOS>", "<EOS>"})
        token_to_id = {t: i for i, t in enumerate(ordered)}
        id_to_token = {i: t for t, i in token_to_id.items()}
        return TextVocab(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, text: str, max_len: int) -> list[int]:
        toks = []
        for tok in _tokenize_text(text):
            toks.append(tok if tok in DOMAIN_TAGS.values() else tok.lower())
        ids = [self.token_to_id.get(t, self.token_to_id["<UNK>"]) for t in toks]
        ids = [self.token_to_id["<BOS>"]] + ids[: max_len - 2] + [self.token_to_id["<EOS>"]]
        if len(ids) < max_len:
            ids += [self.token_to_id["<PAD>"]] * (max_len - len(ids))
        return ids


def _schema_program() -> Program:
    insts = [
        Instruction(opcode="EXTRACT_STR", args={"key": "name"}, dest="name"),
        Instruction(opcode="EXTRACT_INT", args={"key": "age"}, dest="age"),
        Instruction(opcode="EXTRACT_STR", args={"key": "city"}, dest="city"),
        Instruction(opcode="EMIT", args={"schema": "person", "fields": {"name": "name", "age": "age", "city": "city"}}),
    ]
    return Program(insts)


def _find_field(text: str, key: str) -> str | None:
    pattern = rf"{re.escape(key)}\s*(?:[:=]|is)\s*([^\n;,.]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _schema_y_from_text(text: str) -> dict[str, Any]:
    name = _find_field(text, "name") or NAMES[0]
    age_val = _find_field(text, "age")
    city = _find_field(text, "city") or CITIES[0]
    age = 30
    if age_val:
        digits = re.findall(r"-?\d+", age_val)
        if digits:
            age = int(digits[0])
    return {"name": name, "age": age, "city": city}


def _math_y_from_text(text: str) -> Any:
    val = eval_math_expression(text)
    if val is not None:
        return val
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    if len(nums) < 2:
        nums = [1, 1]
    if re.search(r"\+|plus|add", text, flags=re.IGNORECASE):
        return nums[0] + nums[1]
    if re.search(r"-|minus|subtract", text, flags=re.IGNORECASE):
        return nums[0] - nums[1]
    if re.search(r"\*|times|multiply", text, flags=re.IGNORECASE):
        return nums[0] * nums[1]
    if re.search(r"/|divided", text, flags=re.IGNORECASE):
        if nums[1] == 0:
            return float("nan")
        return nums[0] / nums[1]
    return nums[0] + nums[1]


def _parse_csp_tasks(text: str) -> dict[str, int]:
    tasks: dict[str, int] = {}
    for name, dur in re.findall(r"([A-Z])\s*=\s*(\d+)", text):
        tasks[name] = int(dur)
    return tasks


def _parse_csp_constraints(text: str) -> list[tuple[str, str]]:
    return re.findall(r"([A-Z])\s*<\s*([A-Z])", text)


def _solve_csp(tasks: dict[str, int], constraints: list[tuple[str, str]]) -> tuple[dict[str, int], bool]:
    preds: dict[str, set[str]] = {t: set() for t in tasks}
    succs: dict[str, set[str]] = {t: set() for t in tasks}
    for a, b in constraints:
        if a in tasks and b in tasks:
            preds[b].add(a)
            succs[a].add(b)
    queue = [t for t, ps in preds.items() if not ps]
    order: list[str] = []
    while queue:
        node = sorted(queue)[0]
        queue.remove(node)
        order.append(node)
        for nxt in list(succs[node]):
            preds[nxt].discard(node)
            if not preds[nxt]:
                queue.append(nxt)
    if len(order) != len(tasks):
        return {}, False
    schedule: dict[str, int] = {}
    t = 0
    for task in order:
        schedule[task] = t
        t += tasks[task]
    return schedule, True


def _csp_y_from_text(text: str) -> dict[str, Any]:
    tasks = _parse_csp_tasks(text)
    constraints = _parse_csp_constraints(text)
    schedule, ok = _solve_csp(tasks, constraints)
    return {"schedule": schedule, "status": "ok" if ok else "infeasible"}


def _flip_y(domain: str, text: str) -> Any:
    if domain == "schema":
        return _canon_y(_schema_y_from_text(text))
    if domain == "math":
        return _canon_y(_math_y_from_text(text))
    return _canon_y(_csp_y_from_text(text))


def _select_texts(texts: list[str], count: int | None) -> list[str]:
    if count is None:
        return texts
    if count <= 0:
        return []
    if not texts:
        return []
    return [texts[i % len(texts)] for i in range(count)]


def _tag_variants(domain: str, variants: list[Variant]) -> list[Variant]:
    return [variant.model_copy(update={"x": apply_domain_tag(domain, variant.x)}) for variant in variants]


def _make_orbits(domain: str, base_x: str, base_y: Any, count: int | None) -> list[Variant]:
    orbit_texts = generate_orbits(domain, base_x)
    if not orbit_texts:
        orbit_texts = [base_x]
    orbit_texts = _select_texts(orbit_texts, count)
    if count is not None and count > 0 and not orbit_texts:
        orbit_texts = [base_x for _ in range(count)]
    base_y = _canon_y(base_y)
    variants = [Variant(x=txt, y=base_y) for txt in orbit_texts]
    return _tag_variants(domain, variants)


def _fallback_math_flip(base_x: str, base_y: Any) -> Variant:
    base_y = _canon_y(base_y)
    nums = [int(x) for x in re.findall(r"-?\d+", base_x)]
    if len(nums) < 2:
        nums = [1, 1]
    if re.search(r"\+|plus|add", base_x, flags=re.IGNORECASE):
        op = "+"
    elif re.search(r"-|minus|subtract", base_x, flags=re.IGNORECASE):
        op = "-"
    elif re.search(r"\*|times|multiply", base_x, flags=re.IGNORECASE):
        op = "*"
    else:
        op = "/"
    nums[0] += 1
    a, b = nums[0], nums[1]
    if op == "+":
        y = a + b
    elif op == "-":
        y = a - b
    elif op == "*":
        y = a * b
    else:
        if b == 0:
            b = 1
        y = a / b
    y = _canon_y(y)
    if outputs_equivalent(y, base_y):
        a += 1
        if op == "+":
            y = a + b
        elif op == "-":
            y = a - b
        elif op == "*":
            y = a * b
        else:
            y = a / b
        y = _canon_y(y)
    x = f"Compute: {a} {op} {b}."
    return Variant(x=x, y=y)


def _fallback_schema_flip(base_y: Any) -> Variant:
    base_y = _canon_y(base_y)
    if isinstance(base_y, dict):
        name = str(base_y.get("name", NAMES[0]))
        age_val = base_y.get("age", 30)
        try:
            age = int(age_val)
        except (TypeError, ValueError):
            age = 30
        city = str(base_y.get("city", CITIES[0]))
    else:
        name = NAMES[0]
        age = 30
        city = CITIES[0]
    new_age = age + 1
    y = {"name": name, "age": new_age, "city": city}
    if outputs_equivalent(y, base_y):
        new_name = _next_name(name)
        y = {"name": new_name, "age": age, "city": city}
    y = _canon_y(y)
    x = f"name={y['name']}; age={y['age']}; city={y['city']}"
    return Variant(x=x, y=y)


def _fallback_csp_flip(base_x: str, base_y: Any) -> Variant:
    base_y = _canon_y(base_y)
    tasks = _parse_csp_tasks(base_x)
    if not tasks:
        tasks = {t: 1 for t in TASKS[:3]}
    constraints = _parse_csp_constraints(base_x)
    if not constraints and len(tasks) >= 2:
        keys = list(tasks.keys())
        constraints = [(keys[0], keys[1])]
    keys = list(tasks.keys())
    if keys:
        tasks[keys[0]] = tasks[keys[0]] + 1
    schedule, ok = _solve_csp(tasks, constraints)
    y = {"schedule": schedule, "status": "ok" if ok else "infeasible"}
    if outputs_equivalent(y, base_y):
        y = {"schedule": {}, "status": "infeasible"}
    y = _canon_y(y)
    task_part = ",".join(f"{t}={tasks[t]}" for t in tasks)
    cons_part = ",".join(f"{a}<{b}" for a, b in constraints)
    x = f"Tasks: {task_part}. Constraints: {cons_part}."
    return Variant(x=x, y=y)


def _make_flips(domain: str, base_x: str, base_y: Any, count: int | None) -> list[Variant]:
    flip_texts = generate_flips(domain, base_x)
    filtered: list[Variant] = []
    base_y = _canon_y(base_y)
    for txt in flip_texts:
        y = _flip_y(domain, txt)
        if not outputs_equivalent(y, base_y):
            filtered.append(Variant(x=txt, y=y))
    if (count is None or count > 0) and not filtered:
        if domain == "schema":
            filtered.append(_fallback_schema_flip(base_y))
        elif domain == "math":
            filtered.append(_fallback_math_flip(base_x, base_y))
        elif domain == "csp":
            filtered.append(_fallback_csp_flip(base_x, base_y))
    if count is None:
        return _tag_variants(domain, filtered)
    if count <= 0:
        return []
    if not filtered:
        return []
    while len(filtered) < count:
        filtered.append(filtered[len(filtered) % len(filtered)])
    return _tag_variants(domain, filtered[:count])


def generate_schema_example(idx: int, rng: random.Random, orbits: int | None, flips: int | None) -> Example:
    name = rng.choice(NAMES)
    age = rng.randint(18, 60)
    city = rng.choice(CITIES)
    note = f"ref{rng.randint(100000, 999999)}"
    templates = [
        "Record: name={name}; age={age}; city={city}; note={note}.",
        "Profile: name={name}, city={city}, age={age}. Note={note}.",
        "name={name}; age={age}; city={city}; note={note}",
        "- name: {name}\n- age: {age}\n- city: {city}\n- note: {note}",
    ]
    x = rng.choice(templates).format(name=name, age=age, city=city, note=note)
    x = apply_domain_tag("schema", x)
    y = numcanon.canon_json({"name": name, "age": age, "city": city})
    prog = _schema_program()
    orbits_out = _make_orbits("schema", x, y, orbits)
    flips_out = _make_flips("schema", x, y, flips)
    constraints = [
        ConstraintSpec(
            id="json_valid",
            type="schema",
            args={"schema": "person", "required": ["name", "age", "city"], "no_extra": True},
        )
    ]
    tag = DOMAIN_TAGS["schema"]
    return Example(
        id=f"schema_{idx}",
        domain="schema",
        domain_tag=tag,
        x=x,
        y=y,
        constraints=constraints,
        proof=program_to_proof(prog),
        orbit=orbits_out,
        flips=flips_out,
    )


_MATH_LEAF_NAMES = ["a", "b", "c", "d", "e"]
_MATH_TEMP_NAMES = ["t0", "t1", "t2"]
_MATH_OP_WORDS = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}


def _math_leaf(idx: int, value: int | float) -> dict[str, Any]:
    return {"kind": "leaf", "idx": idx, "value": value}


def _math_node(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "op", "op": op, "left": left, "right": right}


def _math_expr_eval(expr: dict[str, Any]) -> float:
    if expr["kind"] == "leaf":
        return float(expr["value"])
    left = _math_expr_eval(expr["left"])
    right = _math_expr_eval(expr["right"])
    op = expr["op"]
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    raise ValueError(f"unsupported op: {op}")


def _math_expr_ops(expr: dict[str, Any], ops: list[str]) -> None:
    if expr["kind"] == "leaf":
        return
    _math_expr_ops(expr["left"], ops)
    ops.append(expr["op"])
    _math_expr_ops(expr["right"], ops)


def _math_expr_text(expr: dict[str, Any], rng: random.Random, root: bool = True) -> str:
    if expr["kind"] == "leaf":
        val = expr["value"]
        if isinstance(val, (int, float)) and val < 0:
            return f"({val})"
        return str(val)
    left = _math_expr_text(expr["left"], rng, root=False)
    right = _math_expr_text(expr["right"], rng, root=False)
    op = expr["op"]
    token = op if rng.random() < 0.6 else _MATH_OP_WORDS[op]
    expr_text = f"{left} {token} {right}"
    if root:
        if rng.random() < 0.3:
            return f"({expr_text})"
        return expr_text
    return f"({expr_text})"


def _math_compile_expr(
    expr: dict[str, Any],
    instructions: list[Instruction],
    temp_idx: list[int],
    is_root: bool = False,
) -> str:
    if expr["kind"] == "leaf":
        return _MATH_LEAF_NAMES[expr["idx"]]
    left_name = _math_compile_expr(expr["left"], instructions, temp_idx, is_root=False)
    right_name = _math_compile_expr(expr["right"], instructions, temp_idx, is_root=False)
    dest = "result" if is_root else _MATH_TEMP_NAMES[temp_idx[0]]
    temp_idx[0] += 1
    instructions.append(
        Instruction(opcode="APPLY_ARITH", args={"a": left_name, "b": right_name, "op": expr["op"]}, dest=dest)
    )
    return dest


def _math_program_from_expr(expr: dict[str, Any]) -> Program:
    ops: list[str] = []
    _math_expr_ops(expr, ops)
    leaves = _math_collect_leaves(expr)
    def _leaf_is_float(value: Any) -> bool:
        if isinstance(value, float):
            return True
        if isinstance(value, Fraction):
            return value.denominator != 1
        return False

    use_float = "/" in ops or any(_leaf_is_float(leaf["value"]) for leaf in leaves)
    extract_op = "EXTRACT_FLOAT" if use_float else "EXTRACT_INT"
    max_idx = max(expr["idx"] for expr in leaves)
    if max_idx + 1 > len(_MATH_LEAF_NAMES):
        raise ValueError("math expression too large for leaf names")
    insts: list[Instruction] = []
    for idx in range(max_idx + 1):
        insts.append(Instruction(opcode=extract_op, args={"index": idx}, dest=_MATH_LEAF_NAMES[idx]))
    temp_idx = [0]
    _math_compile_expr(expr, insts, temp_idx, is_root=True)
    insts.append(Instruction(opcode="EMIT_NUM", args={"value": "result"}))
    return Program(insts)


def math_prompt_to_proof_tokens(text: str) -> list[str] | None:
    expr = parse_math_expression_ast(text)
    if expr is None:
        return None
    try:
        program = _math_program_from_expr(expr)
    except ValueError:
        return None
    return program_to_tokens(program)


def _math_collect_leaves(expr: dict[str, Any]) -> list[dict[str, Any]]:
    if expr["kind"] == "leaf":
        return [expr]
    return _math_collect_leaves(expr["left"]) + _math_collect_leaves(expr["right"])


def _math_build_expr(values: list[int | float], ops: list[str], grouping: str) -> dict[str, Any]:
    leaves = [_math_leaf(i, values[i]) for i in range(len(values))]
    if len(values) == 2:
        return _math_node(ops[0], leaves[0], leaves[1])
    if len(values) == 3:
        if grouping == "right":
            right = _math_node(ops[1], leaves[1], leaves[2])
            return _math_node(ops[0], leaves[0], right)
        left = _math_node(ops[0], leaves[0], leaves[1])
        return _math_node(ops[1], left, leaves[2])
    if len(values) == 4:
        if grouping == "right":
            right = _math_node(ops[2], leaves[2], leaves[3])
            mid = _math_node(ops[1], leaves[1], right)
            return _math_node(ops[0], leaves[0], mid)
        if grouping == "pair":
            left = _math_node(ops[0], leaves[0], leaves[1])
            right = _math_node(ops[2], leaves[2], leaves[3])
            return _math_node(ops[1], left, right)
        left = _math_node(ops[0], leaves[0], leaves[1])
        mid = _math_node(ops[1], left, leaves[2])
        return _math_node(ops[2], mid, leaves[3])
    if len(values) == 5:
        if grouping == "right":
            right = _math_node(ops[3], leaves[3], leaves[4])
            mid = _math_node(ops[2], leaves[2], right)
            mid2 = _math_node(ops[1], leaves[1], mid)
            return _math_node(ops[0], leaves[0], mid2)
        if grouping == "pair":
            left = _math_node(ops[0], leaves[0], leaves[1])
            right = _math_node(ops[2], leaves[2], leaves[3])
            right2 = _math_node(ops[3], right, leaves[4])
            return _math_node(ops[1], left, right2)
        if grouping == "center":
            left = _math_node(ops[0], leaves[0], _math_node(ops[1], leaves[1], leaves[2]))
            right = _math_node(ops[3], leaves[3], leaves[4])
            return _math_node(ops[2], left, right)
        left = _math_node(ops[0], leaves[0], leaves[1])
        mid = _math_node(ops[1], left, leaves[2])
        mid2 = _math_node(ops[2], mid, leaves[3])
        return _math_node(ops[3], mid2, leaves[4])
    raise ValueError("unsupported expression size")


def _sample_math_value(rng: random.Random, num_min: int, num_max: int, allow_float: bool) -> int | float:
    val = rng.randint(num_min, num_max)
    if allow_float and rng.random() < 0.25:
        frac = rng.choice([0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9])
        sign = -1 if val < 0 else 1
        if val == 0:
            sign = -1 if rng.random() < 0.5 else 1
        val = round(val + sign * frac, 2)
    return val


def generate_math_example(idx: int, rng: random.Random, orbits: int | None, flips: int | None) -> Example:
    num_min, num_max = -9999, 9999
    op_count = rng.choices([2, 3, 4], weights=[0.35, 0.35, 0.3], k=1)[0]
    values: list[int | float] = []
    ops: list[str] = []
    while True:
        values = []
        ops = rng.choices(["+", "-", "*", "/"], weights=[0.35, 0.25, 0.25, 0.15], k=op_count)
        avoid_zero = "/" in ops
        allow_float = "/" in ops and rng.random() < 0.5
        for _ in range(op_count + 1):
            val = _sample_math_value(rng, num_min, num_max, allow_float)
            while avoid_zero and abs(val) < 1e-9:
                val = _sample_math_value(rng, num_min, num_max, allow_float)
            values.append(val)
        if op_count == 2:
            grouping = rng.choice(["left", "right"])
        elif op_count == 3:
            grouping = rng.choice(["left", "right", "pair"])
        else:
            grouping = rng.choice(["left", "right", "pair", "center"])
        expr = _math_build_expr(values, ops, grouping)
        try:
            y = _math_expr_eval(expr)
        except ZeroDivisionError:
            continue
        if math.isfinite(y):
            break
    if abs(y - round(y)) <= 1e-9:
        y = int(round(y))
    templates = [
        "Compute: {expr}.",
        "What is {expr}?",
        "Evaluate: {expr}.",
        "Solve: {expr}.",
        "Calculate the value of {expr}.",
        "Find the result of {expr}.",
        "Work out {expr}.",
        "Compute exactly: {expr}.",
    ]
    expr_text = _math_expr_text(expr, rng, root=True)
    x = rng.choice(templates).format(expr=expr_text)
    x = apply_domain_tag("math", x)
    prog = _math_program_from_expr(expr)
    gold_tokens = math_prompt_to_proof_tokens(x)
    y = numcanon.canon_json(y)
    orbits_out = _make_orbits("math", x, y, orbits)
    flips_out = _make_flips("math", x, y, flips)
    constraints = [ConstraintSpec(id="arith", type="arithmetic", args={})]
    tag = DOMAIN_TAGS["math"]
    return Example(
        id=f"math_{idx}",
        domain="math",
        domain_tag=tag,
        x=x,
        y=y,
        constraints=constraints,
        proof=program_to_proof(prog),
        proof_tokens_gold=gold_tokens,
        orbit=orbits_out,
        flips=flips_out,
    )


def _csp_program() -> Program:
    insts = [
        Instruction(opcode="APPLY_TOPO", args={}, dest="order"),
        Instruction(opcode="APPLY_CUMSUM", args={}, dest="schedule"),
        Instruction(opcode="EMIT_SCHEDULE", args={}),
    ]
    return Program(insts)


def generate_csp_example(idx: int, rng: random.Random, orbits: int | None, flips: int | None) -> Example:
    task_count = rng.randint(4, min(12, len(TASKS)))
    tasks = rng.sample(TASKS, k=task_count)
    durations = {t: rng.randint(1, 20) for t in tasks}
    order = list(tasks)
    rng.shuffle(order)
    edge_style = rng.choice(["random", "dense", "sparse", "chain", "fan"])
    if edge_style == "dense":
        edge_prob = rng.uniform(0.45, 0.85)
    elif edge_style == "sparse":
        edge_prob = rng.uniform(0.05, 0.2)
    else:
        edge_prob = rng.uniform(0.15, 0.6)
    cons: list[tuple[str, str]] = []
    if edge_style == "chain":
        for a, b in zip(order, order[1:]):
            cons.append((a, b))
    elif edge_style == "fan":
        root = order[0]
        for b in order[1:]:
            if rng.random() < edge_prob:
                cons.append((root, b))
    else:
        for i in range(task_count):
            for j in range(i + 1, task_count):
                if rng.random() < edge_prob:
                    cons.append((order[i], order[j]))
    if not cons and task_count >= 2:
        i = rng.randrange(0, task_count - 1)
        j = rng.randrange(i + 1, task_count)
        cons.append((order[i], order[j]))
    rng.shuffle(cons)
    task_sep = rng.choice([",", ", ", "; ", " | "])
    cons_sep = rng.choice([",", ", ", "; ", " and ", " | "])
    task_part = task_sep.join(f"{t}={durations[t]}" for t in tasks)
    cons_part = cons_sep.join(f"{a}<{b}" for a, b in cons)
    templates = [
        "Tasks: {tasks}. Constraints: {cons}.",
        "Schedule tasks {tasks}. Precedence: {cons}.",
        "Given tasks {tasks}, obey constraints {cons}.",
        "Tasks (durations): {tasks}. Must satisfy: {cons}.",
        "Plan tasks {tasks} with constraints {cons}.",
    ]
    x = rng.choice(templates).format(tasks=task_part, cons=cons_part)
    x = apply_domain_tag("csp", x)
    prog = _csp_program()
    proof = program_to_proof(prog)
    constraints = [ConstraintSpec(id="csp", type="csp", args={"tasks": durations, "constraints": cons})]
    y = runtime_solve(x, [c.model_dump() for c in constraints], proof.get("tokens", []))
    if isinstance(y, dict) and y.get("status") == "unsat":
        y = {**y, "status": "infeasible"}
    y = numcanon.canon_json(y)
    orbits_out = _make_orbits("csp", x, y, orbits)
    flips_out = _make_flips("csp", x, y, flips)
    tag = DOMAIN_TAGS["csp"]
    return Example(
        id=f"csp_{idx}",
        domain="csp",
        domain_tag=tag,
        x=x,
        y=y,
        constraints=constraints,
        proof=proof,
        orbit=orbits_out,
        flips=flips_out,
    )


def _example_signature(ex: Example) -> str:
    payload = {
        "domain": ex.domain,
        "x": ex.x,
        "y": ex.y,
        "constraints": [c.model_dump() for c in ex.constraints],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def generate_dataset(
    domain: str,
    n: int,
    seed: int,
    orbits: int | None = None,
    flips: int | None = None,
    max_attempts_per_row: int = 50,
) -> list[Example]:
    rng = random.Random(seed)
    random.seed(seed)
    examples: list[Example] = []
    seen_sigs: set[str] = set()
    duplicates = 0
    attempts = 0
    gen_idx = 0
    max_total_attempts = max_attempts_per_row * max(n, 1)
    recent_attempts: list[int] = []
    collision_counts: dict[str, int] = {}
    while len(examples) < n and attempts < max_total_attempts:
        if domain == "schema":
            ex = generate_schema_example(gen_idx, rng, orbits, flips)
        elif domain == "math":
            ex = generate_math_example(gen_idx, rng, orbits, flips)
        elif domain == "csp":
            ex = generate_csp_example(gen_idx, rng, orbits, flips)
        else:
            raise ValueError(f"Unknown domain: {domain}")
        gen_idx += 1
        attempts += 1
        sig = _example_signature(ex)
        if sig in seen_sigs:
            duplicates += 1
            collision_counts[sig] = collision_counts.get(sig, 0) + 1
            recent_attempts.append(1)
            if len(recent_attempts) > 1000:
                recent_attempts.pop(0)
            continue
        seen_sigs.add(sig)
        examples.append(ex)
        recent_attempts.append(0)
        if len(recent_attempts) > 1000:
            recent_attempts.pop(0)
        if attempts % 1000 == 0:
            window = max(1, len(recent_attempts))
            collision_rate = sum(recent_attempts) / float(window)
            top = sorted(collision_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_repr = []
            for sig_val, count in top:
                sig_hash = hashlib.sha1(sig_val.encode("utf-8")).hexdigest()[:8]
                top_repr.append(f"{sig_hash}:{count}")
            top_str = ",".join(top_repr)
            print(
                f"data_gen progress domain={domain} attempts={attempts} uniques={len(examples)} "
                f"collision_rate_last_1k={collision_rate:.3f} top_collisions={top_str}"
            )
    if len(examples) < n:
        print(
            f"data_gen warning domain={domain} requested={n} generated={len(examples)} "
            f"duplicates={duplicates} attempts={attempts}"
        )
    window = max(1, len(recent_attempts))
    collision_rate = sum(recent_attempts) / float(window)
    top = sorted(collision_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_repr = []
    for sig_val, count in top:
        sig_hash = hashlib.sha1(sig_val.encode("utf-8")).hexdigest()[:8]
        top_repr.append(f"{sig_hash}:{count}")
    top_str = ",".join(top_repr)
    print(
        f"data_gen summary domain={domain} attempts={attempts} uniques={len(examples)} "
        f"collision_rate_last_1k={collision_rate:.3f} top_collisions={top_str}"
    )
    return examples


def save_dataset(path: str, examples: list[Example]) -> None:
    write_jsonl(path, [ex.model_dump() for ex in examples])


def load_dataset(path: str) -> list[Example]:
    return [Example.model_validate(row) for row in read_jsonl(path)]


def load_dataset_with_variants(path: str) -> list[Example]:
    examples = load_dataset(path)
    expanded: list[Example] = []
    for ex in examples:
        expanded.append(ex)
        for i, orbit in enumerate(ex.orbit):
            expanded.append(
                ex.model_copy(
                    update={"id": f"{ex.id}:orbit:{i}", "x": orbit.x, "y": orbit.y or ex.y, "orbit": [], "flips": []}
                )
            )
        for i, flip in enumerate(ex.flips):
            expanded.append(
                ex.model_copy(update={"id": f"{ex.id}:flip:{i}", "x": flip.x, "y": flip.y or ex.y, "orbit": [], "flips": []})
            )
    return expanded


def build_program_vocab() -> Any:
    return build_default_vocab()


def _collect_proof_tokens(
    proof: dict[str, Any] | None,
    vocab: TokenVocab,
    seen: set[str],
    missing_counts: Counter[str],
    missing_id_counts: Counter[int],
) -> None:
    if not proof:
        return
    if isinstance(proof, dict) and proof.get("dsl") == "PTv1":
        tokens = proof.get("tokens", [])
        for tok in tokens:
            if isinstance(tok, int):
                decoded = vocab.decode(tok)
                if decoded == "<UNK>":
            missing_id_counts[tok] += 1


def _select_proof_tokens(ex: Example | dict[str, Any], source: str) -> dict[str, Any] | None:
    if source not in {"proof", "gold", "prefer_gold"}:
        raise ValueError(f"Unknown proof source: {source}")
    proof = ex.proof if isinstance(ex, Example) else ex.get("proof")
    if source in {"gold", "prefer_gold"}:
        gold = ex.proof_tokens_gold if isinstance(ex, Example) else ex.get("proof_tokens_gold")
        if isinstance(gold, list) and gold:
            return {"dsl": "PTv1", "tokens": gold}
    return proof
                else:
                    seen.add(decoded)
                continue
            tok_str = str(tok)
            seen.add(tok_str)
            if tok_str not in vocab.token_to_id:
                missing_counts[tok_str] += 1
        return
    try:
        program = Program.from_dict(proof) if isinstance(proof, dict) else None
    except Exception as exc:
        missing_counts[f"invalid_proof:{exc}"] += 1
        return
    if program is None:
        return
    for tok_str in program_to_tokens(program):
        seen.add(tok_str)
        if tok_str not in vocab.token_to_id:
            missing_counts[tok_str] += 1


def audit_proof_tokens(
    examples: Iterable[Example] | Iterable[dict[str, Any]],
    *,
    proof_source: str = "proof",
) -> list[str]:
    vocab = build_default_vocab()
    seen: set[str] = set()
    missing_counts: Counter[str] = Counter()
    missing_id_counts: Counter[int] = Counter()
    for ex in examples:
        proof = _select_proof_tokens(ex, proof_source)
        _collect_proof_tokens(proof, vocab, seen, missing_counts, missing_id_counts)
    if missing_counts or missing_id_counts:
        sample_missing = missing_counts.most_common(10)
        sample_ids = missing_id_counts.most_common(10)
        msg = "Unknown proof tokens detected"
        if missing_counts:
            msg += f" tokens={sample_missing}"
        if missing_id_counts:
            msg += f" token_ids={sample_ids}"
        msg += f" missing_count={sum(missing_counts.values())} missing_id_count={sum(missing_id_counts.values())}"
        raise ValueError(msg)
    return sorted(seen)


def audit_proof_tokens_from_paths(paths: Iterable[str], *, proof_source: str = "proof") -> list[str]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(list(read_jsonl(path)))
    return audit_proof_tokens(rows, proof_source=proof_source)


def build_program_vocab_from_examples(
    examples: Iterable[Example],
    *,
    proof_source: str = "proof",
) -> TokenVocab:
    tokens = audit_proof_tokens(examples, proof_source=proof_source)
    return build_default_vocab(extra_tokens=tokens)


def _hash_tokens(tokens: list[Any]) -> str:
    payload = json.dumps(tokens, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def program_to_proof(program: Program) -> dict[str, Any]:
    tokens = program_to_tokens(program)
    return {"dsl": "PTv1", "tokens": tokens, "sha256": _hash_tokens(tokens), "aux": {}}


def proof_to_token_ids(proof: dict[str, Any], vocab: Any, strict: bool = True) -> list[int]:
    if not proof:
        return []
    if proof.get("dsl") == "PTv1":
        tokens = proof.get("tokens", [])
        ids: list[int] = []
        for tok in tokens:
            if isinstance(tok, int):
                if strict and tok not in vocab.id_to_token:
                    raise ValueError(f"Unknown proof token id: {tok}")
                ids.append(tok)
            else:
                tok_str = str(tok)
                if strict and tok_str not in vocab.token_to_id:
                    raise ValueError(f"Unknown proof token: {tok_str}")
                ids.append(vocab.encode(tok_str))
        return ids
    program = Program.from_dict(proof)
    if strict:
        missing = [tok for tok in program_to_tokens(program) if tok not in vocab.token_to_id]
        if missing:
            raise ValueError(f"Unknown proof tokens: {sorted(set(missing))[:10]}")
    return encode_program(program, vocab)


def proof_to_program(proof: dict[str, Any], vocab: Any) -> Program:
    if not proof:
        return Program(instructions=[])
    if proof.get("dsl") == "PTv1":
        token_ids = proof_to_token_ids(proof, vocab)
        return decode_program(token_ids, vocab)
    return Program.from_dict(proof)


def collate_batch(
    batch: list[Example],
    text_vocab: TextVocab,
    prog_vocab: Any,
    max_text_len: int,
    max_prog_len: int,
    *,
    proof_source: str = "proof",
) -> dict[str, Any]:
    inputs = torch.tensor([text_vocab.encode(ex.x, max_text_len) for ex in batch], dtype=torch.long)
    pad_id = prog_vocab.token_to_id["<PAD>"]
    prog_ids = []
    for ex in batch:
        proof = _select_proof_tokens(ex, proof_source)
        proof_ids = proof_to_token_ids(proof, prog_vocab) if proof else []
        if not proof_ids:
            prog_ids.append([pad_id] * max_prog_len)
            continue
        prog_ids.append(proof_ids[:max_prog_len])
    # Pad programs
    prog_ids = [p + [pad_id] * (max_prog_len - len(p)) if len(p) < max_prog_len else p for p in prog_ids]
    programs = torch.tensor(prog_ids, dtype=torch.long)
    return {"input_ids": inputs, "program_ids": programs}
