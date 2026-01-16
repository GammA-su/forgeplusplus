from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from pydantic import BaseModel, Field

from fc.dsl.codec import decode_program, encode_program, program_to_tokens
from fc.dsl.program import Instruction, Program
from fc.dsl.tokens import CITIES, NAMES, TASKS, build_default_vocab
from fc.morph.flips import generate_flips, _next_name
from fc.morph.orbit import generate_orbits
from fc.morph.equiv import outputs_equivalent
from fc.util.jsonl import read_jsonl, write_jsonl
from fc.util.tags import apply_domain_tag


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
    x: str
    y: Any
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    proof: dict[str, Any]
    orbit: list[Variant] = Field(default_factory=list)
    flips: list[Variant] = Field(default_factory=list)


@dataclass(frozen=True)
class TextVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @staticmethod
    def build(texts: Iterable[str]) -> "TextVocab":
        tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        for text in texts:
            for tok in re.findall(r"\b\w+\b|[+\-*/]", text):
                tokens.add(tok.lower())
        ordered = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + sorted(tokens - {"<PAD>", "<UNK>", "<BOS>", "<EOS>"})
        token_to_id = {t: i for i, t in enumerate(ordered)}
        id_to_token = {i: t for t, i in token_to_id.items()}
        return TextVocab(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, text: str, max_len: int) -> list[int]:
        toks = [tok.lower() for tok in re.findall(r"\b\w+\b|[+\-*/]", text)]
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


def _math_y_from_text(text: str) -> float:
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
        return _schema_y_from_text(text)
    if domain == "math":
        return _math_y_from_text(text)
    return _csp_y_from_text(text)


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
    variants = [Variant(x=txt, y=base_y) for txt in orbit_texts]
    return _tag_variants(domain, variants)


def _fallback_math_flip(base_x: str, base_y: Any) -> Variant:
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
    x = f"Compute: {a} {op} {b}."
    return Variant(x=x, y=y)


def _fallback_schema_flip(base_y: Any) -> Variant:
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
    x = f"name={y['name']}; age={y['age']}; city={y['city']}"
    return Variant(x=x, y=y)


def _fallback_csp_flip(base_x: str, base_y: Any) -> Variant:
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
    task_part = ",".join(f"{t}={tasks[t]}" for t in tasks)
    cons_part = ",".join(f"{a}<{b}" for a, b in constraints)
    x = f"Tasks: {task_part}. Constraints: {cons_part}."
    return Variant(x=x, y=y)


def _make_flips(domain: str, base_x: str, base_y: Any, count: int | None) -> list[Variant]:
    flip_texts = generate_flips(domain, base_x)
    filtered: list[Variant] = []
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
    templates = [
        "Record: name={name}; age={age}; city={city}.",
        "Profile: name={name}, city={city}, age={age}.",
        "name={name}; age={age}; city={city}; note=ignore",
        "- name: {name}\n- age: {age}\n- city: {city}",
    ]
    x = rng.choice(templates).format(name=name, age=age, city=city)
    x = apply_domain_tag("schema", x)
    y = {"name": name, "age": age, "city": city}
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
    return Example(
        id=f"schema_{idx}",
        domain="schema",
        x=x,
        y=y,
        constraints=constraints,
        proof=prog.to_dict(),
        orbit=orbits_out,
        flips=flips_out,
    )


def _math_program(op: str) -> Program:
    op_map = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}
    insts = [
        Instruction(opcode="EXTRACT_INT", args={"index": 0}, dest="a"),
        Instruction(opcode="EXTRACT_INT", args={"index": 1}, dest="b"),
        Instruction(opcode=op_map[op], args={"a": "a", "b": "b"}, dest="result"),
        Instruction(opcode="EMIT", args={"schema": "math", "fields": {"result": "result"}}),
    ]
    return Program(insts)


def generate_math_example(idx: int, rng: random.Random, orbits: int | None, flips: int | None) -> Example:
    a = rng.randint(1, 12)
    b = rng.randint(1, 12)
    op = rng.choice(["+", "-", "*", "/"])
    if op == "/":
        a = a * b
    templates = [
        "Compute: {a} {op} {b}.",
        "What is {a} {op_word} {b}?",
    ]
    op_word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]
    x = rng.choice(templates).format(a=a, b=b, op=op, op_word=op_word)
    x = apply_domain_tag("math", x)
    if op == "+":
        y = a + b
    elif op == "-":
        y = a - b
    elif op == "*":
        y = a * b
    else:
        y = a / b
    prog = _math_program(op)
    orbits_out = _make_orbits("math", x, y, orbits)
    flips_out = _make_flips("math", x, y, flips)
    constraints = [ConstraintSpec(id="arith", type="arithmetic", args={"op": op})]
    return Example(
        id=f"math_{idx}",
        domain="math",
        x=x,
        y=y,
        constraints=constraints,
        proof=program_to_proof(prog),
        orbit=orbits_out,
        flips=flips_out,
    )


def _csp_program() -> Program:
    insts = [
        Instruction(opcode="SOLVE_CSP", args={}, dest="schedule"),
        Instruction(opcode="EMIT", args={"schema": "schedule", "fields": {"schedule": "schedule", "status": "status"}}),
    ]
    return Program(insts)


def generate_csp_example(idx: int, rng: random.Random, orbits: int | None, flips: int | None) -> Example:
    tasks = rng.sample(TASKS, k=3)
    durations = {t: rng.randint(1, 3) for t in tasks}
    cons = [(tasks[0], tasks[1]), (tasks[1], tasks[2])]
    task_part = ",".join(f"{t}={durations[t]}" for t in tasks)
    cons_part = ",".join(f"{a}<{b}" for a, b in cons)
    x = f"Tasks: {task_part}. Constraints: {cons_part}."
    x = apply_domain_tag("csp", x)
    # Expected schedule is sequential in order tasks[0], tasks[1], tasks[2]
    schedule = {}
    t = 0
    for task in tasks:
        schedule[task] = t
        t += durations[task]
    y = {"schedule": schedule, "status": "ok"}
    prog = _csp_program()
    orbits_out = _make_orbits("csp", x, y, orbits)
    flips_out = _make_flips("csp", x, y, flips)
    constraints = [ConstraintSpec(id="csp", type="csp", args={"tasks": durations, "constraints": cons})]
    return Example(
        id=f"csp_{idx}",
        domain="csp",
        x=x,
        y=y,
        constraints=constraints,
        proof=program_to_proof(prog),
        orbit=orbits_out,
        flips=flips_out,
    )


def generate_dataset(domain: str, n: int, seed: int, orbits: int | None = None, flips: int | None = None) -> list[Example]:
    rng = random.Random(seed)
    random.seed(seed)
    examples = []
    for i in range(n):
        if domain == "schema":
            examples.append(generate_schema_example(i, rng, orbits, flips))
        elif domain == "math":
            examples.append(generate_math_example(i, rng, orbits, flips))
        elif domain == "csp":
            examples.append(generate_csp_example(i, rng, orbits, flips))
        else:
            raise ValueError(f"Unknown domain: {domain}")
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


def program_to_proof(program: Program) -> dict[str, Any]:
    return {"dsl": "PTv1", "tokens": program_to_tokens(program), "aux": {}}


def proof_to_token_ids(proof: dict[str, Any], vocab: Any) -> list[int]:
    if not proof:
        return []
    if proof.get("dsl") == "PTv1":
        tokens = proof.get("tokens", [])
        ids: list[int] = []
        for tok in tokens:
            if isinstance(tok, int):
                ids.append(tok)
            else:
                ids.append(vocab.encode(str(tok)))
        return ids
    program = Program.from_dict(proof)
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
) -> dict[str, Any]:
    inputs = torch.tensor([text_vocab.encode(ex.x, max_text_len) for ex in batch], dtype=torch.long)
    pad_id = prog_vocab.token_to_id["<PAD>"]
    prog_ids = []
    for ex in batch:
        proof_ids = proof_to_token_ids(ex.proof, prog_vocab)
        if not proof_ids:
            prog_ids.append([pad_id] * max_prog_len)
            continue
        prog_ids.append(proof_ids[:max_prog_len])
    # Pad programs
    prog_ids = [p + [pad_id] * (max_prog_len - len(p)) if len(p) < max_prog_len else p for p in prog_ids]
    programs = torch.tensor(prog_ids, dtype=torch.long)
    return {"input_ids": inputs, "program_ids": programs}
