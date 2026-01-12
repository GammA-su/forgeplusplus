from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from pydantic import BaseModel, Field

from fc.dsl.program import Instruction, Program
from fc.dsl.tokens import CITIES, NAMES, TASKS, build_default_vocab
from fc.morph.flips import generate_flips
from fc.morph.orbit import generate_orbits
from fc.util.jsonl import read_jsonl, write_jsonl


class ConstraintSpec(BaseModel):
    id: str
    type: str
    args: dict[str, Any] = Field(default_factory=dict)


class Variant(BaseModel):
    x: str


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


def generate_schema_example(idx: int, rng: random.Random) -> Example:
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
    y = {"name": name, "age": age, "city": city}
    prog = _schema_program()
    orbits = [Variant(x=t) for t in generate_orbits("schema", x)]
    flips = [Variant(x=t) for t in generate_flips("schema", x)]
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
        orbit=orbits,
        flips=flips,
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


def generate_math_example(idx: int, rng: random.Random) -> Example:
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
    if op == "+":
        y = a + b
    elif op == "-":
        y = a - b
    elif op == "*":
        y = a * b
    else:
        y = a / b
    prog = _math_program(op)
    orbits = [Variant(x=t) for t in generate_orbits("math", x)]
    flips = [Variant(x=t) for t in generate_flips("math", x)]
    constraints = [ConstraintSpec(id="arith", type="arithmetic", args={"op": op})]
    return Example(
        id=f"math_{idx}",
        domain="math",
        x=x,
        y=y,
        constraints=constraints,
        proof=prog.to_dict(),
        orbit=orbits,
        flips=flips,
    )


def _csp_program() -> Program:
    insts = [
        Instruction(opcode="SOLVE_CSP", args={}, dest="schedule"),
        Instruction(opcode="EMIT", args={"schema": "schedule", "fields": {"schedule": "schedule", "status": "status"}}),
    ]
    return Program(insts)


def generate_csp_example(idx: int, rng: random.Random) -> Example:
    tasks = rng.sample(TASKS, k=3)
    durations = {t: rng.randint(1, 3) for t in tasks}
    cons = [(tasks[0], tasks[1]), (tasks[1], tasks[2])]
    task_part = ",".join(f"{t}={durations[t]}" for t in tasks)
    cons_part = ",".join(f"{a}<{b}" for a, b in cons)
    x = f"Tasks: {task_part}. Constraints: {cons_part}."
    # Expected schedule is sequential in order tasks[0], tasks[1], tasks[2]
    schedule = {}
    t = 0
    for task in tasks:
        schedule[task] = t
        t += durations[task]
    y = {"schedule": schedule, "status": "ok"}
    prog = _csp_program()
    orbits = [Variant(x=t) for t in generate_orbits("csp", x)]
    flips = [Variant(x=t) for t in generate_flips("csp", x)]
    constraints = [ConstraintSpec(id="csp", type="csp", args={"tasks": durations, "constraints": cons})]
    return Example(
        id=f"csp_{idx}",
        domain="csp",
        x=x,
        y=y,
        constraints=constraints,
        proof=prog.to_dict(),
        orbit=orbits,
        flips=flips,
    )


def generate_dataset(domain: str, n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    random.seed(seed)
    examples = []
    for i in range(n):
        if domain == "schema":
            examples.append(generate_schema_example(i, rng))
        elif domain == "math":
            examples.append(generate_math_example(i, rng))
        elif domain == "csp":
            examples.append(generate_csp_example(i, rng))
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
                    update={"id": f"{ex.id}:orbit:{i}", "x": orbit.x, "orbit": [], "flips": []}
                )
            )
        for i, flip in enumerate(ex.flips):
            expanded.append(
                ex.model_copy(update={"id": f"{ex.id}:flip:{i}", "x": flip.x, "orbit": [], "flips": []})
            )
    return expanded


def build_program_vocab() -> Any:
    return build_default_vocab()


def collate_batch(
    batch: list[Example],
    text_vocab: TextVocab,
    prog_vocab: Any,
    max_text_len: int,
    max_prog_len: int,
) -> dict[str, Any]:
    inputs = torch.tensor([text_vocab.encode(ex.x, max_text_len) for ex in batch], dtype=torch.long)
    prog_ids = []
    for ex in batch:
        prog = Program.from_dict(ex.proof)
        from fc.dsl.codec import encode_program

        prog_ids.append(encode_program(prog, prog_vocab)[:max_prog_len])
    # Pad programs
    pad_id = prog_vocab.token_to_id["<PAD>"]
    prog_ids = [p + [pad_id] * (max_prog_len - len(p)) if len(p) < max_prog_len else p for p in prog_ids]
    programs = torch.tensor(prog_ids, dtype=torch.long)
    return {"input_ids": inputs, "program_ids": programs}
