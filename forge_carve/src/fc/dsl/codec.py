from __future__ import annotations

from typing import Iterable

from fc.dsl.program import Instruction, Program
from fc.dsl.tokens import OPCODES, TokenVocab, Value


def _value_to_tokens(value: Value) -> list[str]:
    if isinstance(value, bool):
        return ["BOOL:true" if value else "BOOL:false"]
    if isinstance(value, int):
        return [f"INT:{value}"]
    if isinstance(value, float):
        return [f"STR:{value}"]
    if isinstance(value, str):
        return [f"STR:{value}"]
    if isinstance(value, list):
        tokens = ["LIST_START"]
        for i, item in enumerate(value):
            if i > 0:
                tokens.append("SEP")
            tokens.extend(_value_to_tokens(item))
        tokens.append("LIST_END")
        return tokens
    if isinstance(value, dict):
        tokens = ["DICT_START"]
        items = list(value.items())
        for i, (k, v) in enumerate(items):
            if i > 0:
                tokens.append("SEP")
            tokens.extend(_value_to_tokens(str(k)))
            tokens.append("SEP")
            tokens.extend(_value_to_tokens(v))
        tokens.append("DICT_END")
        return tokens
    raise TypeError(f"Unsupported value: {value}")


def program_to_tokens(program: Program) -> list[str]:
    tokens: list[str] = ["<BOS>", "BEGIN"]
    for inst in program.instructions:
        tokens.extend(["OP", inst.opcode])
        if inst.dest is not None:
            tokens.extend(["DEST", f"STR:{inst.dest}"])
        for key in sorted(inst.args.keys()):
            tokens.extend(["ARG", key, "VAL"])
            tokens.extend(_value_to_tokens(inst.args[key]))
    tokens.extend(["END", "<EOS>"])
    return tokens


def encode_program(program: Program, vocab: TokenVocab) -> list[int]:
    tokens = program_to_tokens(program)
    return [vocab.encode(tok) for tok in tokens]


def _parse_value(tokens: list[str], idx: int) -> tuple[Value, int]:
    tok = tokens[idx]
    if tok.startswith("INT:"):
        try:
            return int(tok.split(":", 1)[1]), idx + 1
        except ValueError:
            return tok.split(":", 1)[1], idx + 1
    if tok.startswith("BOOL:"):
        return tok.split(":", 1)[1] == "true", idx + 1
    if tok.startswith("STR:"):
        return tok.split(":", 1)[1], idx + 1
    if tok == "LIST_START":
        items: list[Value] = []
        idx += 1
        while tokens[idx] != "LIST_END":
            if tokens[idx] == "SEP":
                idx += 1
                continue
            item, idx = _parse_value(tokens, idx)
            items.append(item)
        return items, idx + 1
    if tok == "DICT_START":
        data: dict[str, Value] = {}
        idx += 1
        while tokens[idx] != "DICT_END":
            if tokens[idx] == "SEP":
                idx += 1
                continue
            key_val, idx = _parse_value(tokens, idx)
            if tokens[idx] == "SEP":
                idx += 1
            val, idx = _parse_value(tokens, idx)
            data[str(key_val)] = val
            if tokens[idx] == "SEP":
                idx += 1
        return data, idx + 1
    raise ValueError(f"Cannot parse value token: {tok}")


def decode_program(token_ids: Iterable[int], vocab: TokenVocab) -> Program:
    tokens = [vocab.decode(i) for i in token_ids]
    if "BEGIN" not in tokens:
        return Program(instructions=[])
    idx = tokens.index("BEGIN") + 1
    instructions: list[Instruction] = []
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "END":
            break
        if tok != "OP":
            idx += 1
            continue
        if idx + 1 >= len(tokens):
            break
        opcode = tokens[idx + 1]
        idx += 2
        if opcode not in OPCODES:
            while idx < len(tokens) and tokens[idx] not in ("OP", "END"):
                idx += 1
            continue
        dest: str | None = None
        args: dict[str, Value] = {}
        while idx < len(tokens):
            tok = tokens[idx]
            if tok == "OP" or tok == "END":
                break
            if tok == "DEST":
                if idx + 1 >= len(tokens):
                    idx += 1
                    continue
                dest_token = tokens[idx + 1]
                if dest_token.startswith("STR:"):
                    dest = dest_token.split(":", 1)[1]
                idx += 2
                continue
            if tok == "ARG":
                if idx + 2 >= len(tokens):
                    idx += 1
                    continue
                key = tokens[idx + 1]
                if tokens[idx + 2] != "VAL":
                    idx += 1
                    continue
                try:
                    val, next_idx = _parse_value(tokens, idx + 3)
                except (ValueError, IndexError):
                    idx += 1
                    continue
                else:
                    args[key] = val
                    idx = next_idx
                continue
            idx += 1
        instructions.append(Instruction(opcode=opcode, args=args, dest=dest))
    return Program(instructions=instructions)


def alignment_distance(a: Program, b: Program) -> int:
    seq_a = a.skeleton()
    seq_b = b.skeleton()
    dp = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
    for i in range(len(seq_a) + 1):
        dp[i][0] = i
    for j in range(len(seq_b) + 1):
        dp[0][j] = j
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]
