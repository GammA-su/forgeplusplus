from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
from collections import defaultdict, deque


@dataclass(frozen=True)
class Instr:
    op: str
    dest: Optional[str]
    args: Dict[str, Any]


def _parse_value(tok: str) -> Any:
    # token formats: STR:foo, INT:0, FLOAT:9.0, BOOL:true
    if tok.startswith("STR:"):
        return tok[4:]
    if tok.startswith("INT:"):
        return int(tok[4:])
    if tok.startswith("FLOAT:"):
        return float(tok[6:])
    if tok.startswith("BOOL:"):
        v = tok[5:].lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
        raise ValueError(f"bad BOOL token: {tok}")
    # fallback: raw token
    return tok


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
            val_tok = tokens[i]
            i += 1
            args[key] = _parse_value(val_tok)

        out.append(Instr(op=op, dest=dest, args=args))

    return out


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> List[str]:
    return _NUM_RE.findall(text)


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

        for ins in prog:
            op = ins.op
            dest = ins.dest
            args = ins.args

            if op == "EXTRACT_INT":
                nums = _extract_numbers(env["x"])
                idx = int(args.get("index", 0))
                val = int(float(nums[idx]))
                if dest is None:
                    raise ValueError("EXTRACT_INT missing DEST")
                env[dest] = val

            elif op == "EXTRACT_FLOAT":
                nums = _extract_numbers(env["x"])
                idx = int(args.get("index", 0))
                val = float(nums[idx])
                if dest is None:
                    raise ValueError("EXTRACT_FLOAT missing DEST")
                env[dest] = val

            elif op == "APPLY_ARITH":
                # args a,b can be references (STR:a) or literals
                a_ref = args.get("a")
                b_ref = args.get("b")
                op_sym = args.get("op")

                a = env[a_ref] if isinstance(a_ref, str) and a_ref in env else a_ref
                b = env[b_ref] if isinstance(b_ref, str) and b_ref in env else b_ref
                opv = env[op_sym] if isinstance(op_sym, str) and op_sym in env else op_sym

                if opv == "+":
                    r = a + b
                elif opv == "-":
                    r = a - b
                elif opv == "*":
                    r = a * b
                elif opv == "/":
                    r = a / b
                else:
                    raise ValueError(f"unknown arithmetic op: {opv}")

                if dest is None:
                    raise ValueError("APPLY_ARITH missing DEST")
                env[dest] = r

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

            elif op == "EMIT_NUM":
                vref = args.get("value")
                v = env[vref] if isinstance(vref, str) and vref in env else vref
                last_emit = v

            elif op == "EMIT_SCHEDULE":
                sched = env.get("schedule") or {}
                status = env.get("status", "ok")
                last_emit = {"schedule": sched, "status": status}

            else:
                raise ValueError(f"unsupported op: {op}")

        return last_emit
