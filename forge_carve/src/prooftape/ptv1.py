from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple
import re
from collections import defaultdict, deque
import math


@dataclass(frozen=True)
class Instr:
    op: str
    dest: Optional[str]
    args: Dict[str, Any]


_DECIMAL_RE = re.compile(r"^\s*([+-]?)(?:(\d+)(?:\.(\d*))?|\.(\d+))\s*$")


def _decimal_to_fraction(text: str) -> Fraction:
    match = _DECIMAL_RE.match(text)
    if not match:
        raise ValueError(f"bad decimal: {text!r}")
    sign, whole, frac_a, frac_b = match.groups()
    frac = frac_b if frac_b is not None else (frac_a or "")
    whole = whole or "0"
    if not frac:
        val = int(whole)
        return Fraction(-val if sign == "-" else val, 1)
    num = int(f"{whole}{frac}")
    if sign == "-":
        num = -num
    return Fraction(num, 10 ** len(frac))


def _parse_value(tok: str) -> Any:
    # token formats: STR:foo, INT:0, FLOAT:9.0, BOOL:true
    if tok.startswith("STR:"):
        return tok[4:]
    if tok.startswith("INT:"):
        return int(tok[4:])
    if tok.startswith("FLOAT:"):
        return _decimal_to_fraction(tok[6:])
    if tok.startswith("BOOL:"):
        v = tok[5:].lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
        raise ValueError(f"bad BOOL token: {tok}")
    # fallback: raw token
    return tok


def _parse_value_stream(tokens: List[str], idx: int) -> Tuple[Any, int]:
    tok = tokens[idx]
    if tok == "LIST_START":
        items: List[Any] = []
        idx += 1
        while tokens[idx] != "LIST_END":
            if tokens[idx] == "SEP":
                idx += 1
                continue
            item, idx = _parse_value_stream(tokens, idx)
            items.append(item)
        return items, idx + 1
    if tok == "DICT_START":
        data: Dict[str, Any] = {}
        idx += 1
        while tokens[idx] != "DICT_END":
            if tokens[idx] == "SEP":
                idx += 1
                continue
            key_val, idx = _parse_value_stream(tokens, idx)
            if tokens[idx] == "SEP":
                idx += 1
            val, idx = _parse_value_stream(tokens, idx)
            data[str(key_val)] = val
            if tokens[idx] == "SEP":
                idx += 1
        return data, idx + 1
    return _parse_value(tok), idx + 1


def parse_tokens(tokens: List[str]) -> List[Instr]:
    """
    Parse PTv1 token streams of the form:

      <BOS> BEGIN
      OP <OPNAME> [DEST STR:name] { ARG key VAL <VALTOK> }*
      ...
      END <EOS>

    into a list[Instr].
    """
    i = 0
    out: List[Instr] = []
    n = len(tokens)

    def peek() -> str:
        return tokens[i] if i < n else ""

    # skip wrappers
    while i < n and tokens[i] in ("<BOS>", "BEGIN"):
        i += 1

    while i < n:
        t = tokens[i]
        if t in ("END", "<EOS>"):
            break
        if t != "OP":
            i += 1
            continue

        i += 1
        if i >= n:
            raise ValueError("dangling OP")
        op = tokens[i]
        i += 1

        dest: Optional[str] = None
        args: Dict[str, Any] = {}

        # optional DEST STR:...
        if peek() == "DEST":
            i += 1
            if tokens[i] != "STR:" and not tokens[i].startswith("STR:"):
                raise ValueError(f"DEST expects STR:* got {tokens[i]}")
            dest = _parse_value(tokens[i])
            i += 1

        # zero or more ARG key VAL value
        while peek() == "ARG":
            i += 1
            key = tokens[i]
            i += 1
            if tokens[i] != "VAL":
                raise ValueError(f"ARG {key} missing VAL, got {tokens[i]}")
            i += 1
            val, next_idx = _parse_value_stream(tokens, i)
            i = next_idx
            args[key] = val

        out.append(Instr(op=op, dest=dest, args=args))

    return out


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_INT_RE = re.compile(r"-?\d+")


def _extract_numbers(text: str) -> List[str]:
    return _NUM_RE.findall(text)

def _extract_ints(text: str) -> List[str]:
    return _INT_RE.findall(text)


def _normalize_stop(stop: str | None) -> str | None:
    if stop is None:
        return None
    if stop in {"\\n", "newline", "line"}:
        return "\n"
    return stop


def _extract_by_key(text: str, key: str, stop: str | None = None) -> str | None:
    stop = _normalize_stop(stop)
    if stop:
        stop_pat = re.escape(stop)
        pattern = rf"{re.escape(key)}\s*(?:[:=]|is)\s*([^{stop_pat}]+)"
    else:
        pattern = rf"{re.escape(key)}\s*(?:[:=]|is)\s*([^\n;,.]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_int(
    text: str,
    key: str | None,
    index: int | None,
    stop: str | None,
) -> int | None:
    if key is not None:
        val = _extract_by_key(text, key, stop=stop)
        if val is None:
            return None
        nums = _NUM_RE.findall(val)
        if not nums:
            return None
        try:
            frac = _decimal_to_fraction(nums[0])
        except ValueError:
            return None
        if frac.denominator != 1:
            return None
        return int(frac.numerator)
    nums = _NUM_RE.findall(text)
    if index is None or index >= len(nums):
        return None
    try:
        frac = _decimal_to_fraction(nums[index])
    except ValueError:
        return None
    if frac.denominator != 1:
        return None
    return int(frac.numerator)


def _extract_float(
    text: str,
    key: str | None,
    index: int | None,
    stop: str | None,
) -> Fraction | None:
    if key is not None:
        val = _extract_by_key(text, key, stop=stop)
        if val is None:
            return None
        nums = _NUM_RE.findall(val)
        if not nums:
            return None
        try:
            return _decimal_to_fraction(nums[0])
        except ValueError:
            return None
    nums = _NUM_RE.findall(text)
    if index is None or index >= len(nums):
        return None
    try:
        return _decimal_to_fraction(nums[index])
    except ValueError:
        return None


def _extract_str(text: str, key: str | None, stop: str | None) -> str | None:
    if key is not None:
        val = _extract_by_key(text, key, stop=stop)
        if val is None:
            return None
        return val.strip()
    return None


def _coerce_number(v: Any) -> Any:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return v
        return Fraction(v).limit_denominator(10**12)
    if isinstance(v, str):
        try:
            return _decimal_to_fraction(v)
        except ValueError:
            return v
    return v


def _topo_order(tasks: Dict[str, int], edges: List[Tuple[str, str]]) -> Optional[List[str]]:
    """
    Deterministic Kahn topo sort.
    Tie-break: insertion order of `tasks` dict (JSON preserves object order).
    Return None on cycle.
    """
    nodes = list(tasks.keys())
    rank = {k: idx for idx, k in enumerate(nodes)}

    adj: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = {k: 0 for k in nodes}

    for u, v in edges:
        if u not in indeg or v not in indeg:
            # ignore unknowns (or raise if you prefer strict)
            continue
        adj[u].append(v)
        indeg[v] += 1

    q = [k for k in nodes if indeg[k] == 0]
    q.sort(key=lambda k: rank[k])

    out: List[str] = []
    while q:
        u = q.pop(0)
        out.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
        q.sort(key=lambda k: rank[k])

    if len(out) != len(nodes):
        return None
    return out


class PTv1Runtime:
    """
    Minimal runtime for the ops shown in your dataset.
    """
    def __init__(self) -> None:
        pass

    def run(self, x: str, constraints: List[dict], tokens: List[str]) -> Any:
        """
        `constraints` is the record["constraints"] list (typed).
        We seed env with:
          - x (string)
          - tasks (dict) and precedence edges for CSP if present
        """
        env: Dict[str, Any] = {"x": x, "status": "ok"}

        # seed typed constraints if present
        for c in constraints or []:
            if c.get("type") == "csp":
                args = c.get("args") or {}
                env["tasks"] = args.get("tasks") or {}
                env["precedence"] = args.get("constraints") or []  # list[list[str,str]]
                break

        prog = parse_tokens(tokens)
        last_emit: Any = None
        last_value: Any = None

        for ins in prog:
            op = ins.op
            dest = ins.dest
            args = ins.args

            if op == "EXTRACT_INT":
                key = args.get("key")
                index = args.get("index")
                stop = args.get("stop")
                idx = int(index) if index is not None else None
                val = _extract_int(env["x"], key, idx, stop)
                if dest is None:
                    raise ValueError("EXTRACT_INT missing DEST")
                if val is None:
                    raise ValueError("EXTRACT_INT missing value")
                env[dest] = val
                last_value = val

            elif op == "EXTRACT_FLOAT":
                key = args.get("key")
                index = args.get("index")
                stop = args.get("stop")
                idx = int(index) if index is not None else None
                val = _extract_float(env["x"], key, idx, stop)
                if dest is None:
                    raise ValueError("EXTRACT_FLOAT missing DEST")
                if val is None:
                    raise ValueError("EXTRACT_FLOAT missing value")
                env[dest] = val
                last_value = val

            elif op == "EXTRACT_STR":
                key = args.get("key")
                stop = args.get("stop")
                val = _extract_str(env["x"], key, stop)
                if dest is None:
                    raise ValueError("EXTRACT_STR missing DEST")
                if val is None:
                    raise ValueError("EXTRACT_STR missing value")
                env[dest] = val
                last_value = val

            elif op == "APPLY_ARITH":
                # args a,b can be references (STR:a) or literals
                a_ref = args.get("a")
                b_ref = args.get("b")
                op_sym = args.get("op")

                a = env[a_ref] if isinstance(a_ref, str) and a_ref in env else a_ref
                b = env[b_ref] if isinstance(b_ref, str) and b_ref in env else b_ref
                opv = env[op_sym] if isinstance(op_sym, str) and op_sym in env else op_sym
                a = _coerce_number(a)
                b = _coerce_number(b)

                opv = str(opv).lower()
                if opv in {"+", "add", "plus"}:
                    r = a + b
                elif opv in {"-", "sub", "minus"}:
                    r = a - b
                elif opv in {"*", "mul", "times", "x"}:
                    r = a * b
                elif opv in {"/", "div", "divide"}:
                    if b == 0:
                        r = float("nan")
                    elif isinstance(a, (int, Fraction)) and isinstance(b, (int, Fraction)):
                        r = Fraction(a, b)
                    else:
                        r = a / b
                else:
                    raise ValueError(f"unknown arithmetic op: {opv}")

                if isinstance(r, Fraction) and r.denominator == 1:
                    r = int(r.numerator)
                elif isinstance(r, float) and r.is_integer():
                    r = int(r)

                if dest is None:
                    raise ValueError("APPLY_ARITH missing DEST")
                env[dest] = r
                last_value = r

            elif op == "APPLY_TOPO":
                tasks = env.get("tasks") or {}
                edges = env.get("precedence") or []
                edges_t = [(u, v) for (u, v) in edges]
                order = _topo_order(tasks, edges_t)
                if order is None:
                    env["status"] = "unsat"
                    order = []
                if dest is None:
                    raise ValueError("APPLY_TOPO missing DEST")
                env[dest] = order
                last_value = order

            elif op == "APPLY_CUMSUM":
                if env.get("status") != "ok":
                    # keep schedule empty in UNSAT
                    sched: Dict[str, int] = {}
                else:
                    tasks = env.get("tasks") or {}
                    order_ref = env.get("order")  # typical DEST is "order"
                    order = order_ref if isinstance(order_ref, list) else []
                    t = 0
                    sched = {}
                    for k in order:
                        sched[k] = t
                        t += int(tasks[k])
                if dest is None:
                    raise ValueError("APPLY_CUMSUM missing DEST")
                env[dest] = sched
                last_value = sched

            elif op == "EMIT_NUM":
                vref = args.get("value")
                v = env[vref] if isinstance(vref, str) and vref in env else vref
                last_emit = v
                last_value = v

            elif op == "EMIT_SCHEDULE":
                sched = env.get("schedule") or {}
                status = env.get("status", "ok")
                last_emit = {"schedule": sched, "status": status}
                last_value = last_emit

            elif op == "EMIT":
                fields = args.get("fields", {})
                out: Dict[str, Any] = {}
                for key, val in fields.items():
                    if isinstance(val, str) and val in env:
                        out[key] = env[val]
                    else:
                        out[key] = val
                last_emit = out
                last_value = out

            else:
                raise ValueError(f"unsupported op: {op}")

        if last_emit is not None:
            return last_emit
        return last_value
